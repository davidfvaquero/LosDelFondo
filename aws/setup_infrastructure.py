#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, WaiterError


BASE_DIR = Path(__file__).resolve().parents[1]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "deportedata")
STACK_PREFIX = PROJECT_NAME.replace("_", "-")
INSTANCE_NAME = os.environ.get("INSTANCE_NAME", f"{STACK_PREFIX}-backend")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "t3.large")
DB_INSTANCE_ID = os.environ.get("DB_INSTANCE_ID", f"{STACK_PREFIX}-rds")
DB_ENGINE_VERSION = os.environ.get("DB_ENGINE_VERSION", "16.13")
DB_NAME = os.environ.get("DB_NAME", "deportedata")
DB_USER = os.environ.get("DB_USER", "deporteadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD") or f"Depo-{secrets.token_hex(8)}"
PARAMETER_PREFIX = os.environ.get("SSM_PARAMETER_PREFIX", f"/{STACK_PREFIX}")
SOURCE_ARCHIVE_KEY = os.environ.get("SOURCE_ARCHIVE_KEY", "releases/current.tar.gz")
ENABLE_EIP = os.environ.get("ENABLE_EIP", "true").lower() == "true"
REPLACE_INSTANCE = os.environ.get("REPLACE_INSTANCE", "false").lower() == "true"
KEY_NAME_OVERRIDE = os.environ.get("AWS_KEY_PAIR_NAME", "")
INSTANCE_PROFILE_NAME = os.environ.get("INSTANCE_PROFILE_NAME", "LabInstanceProfile")
APP_DIR = f"/opt/{STACK_PREFIX}"
COMPOSE_URL = "https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64"
COMPOSE_BIN = "/usr/local/bin/docker-compose"
REFRESH_SCRIPT_PATH = f"/usr/local/bin/{STACK_PREFIX}-refresh.sh"
SYSTEMD_SERVICE_PATH = f"/etc/systemd/system/{STACK_PREFIX}.service"
SYSTEMD_SERVICE_NAME = f"{STACK_PREFIX}.service"

session = boto3.Session(region_name=REGION)
ec2 = session.client("ec2")
rds = session.client("rds")
s3 = session.client("s3")
ssm = session.client("ssm")
logs = session.client("logs")
cloudwatch = session.client("cloudwatch")
sts = session.client("sts")

ACCOUNT_ID = sts.get_caller_identity()["Account"]
S3_BUCKET = os.environ.get("S3_BUCKET", f"{STACK_PREFIX}-{ACCOUNT_ID}-{REGION}")
OUTPUT_FILE = BASE_DIR / "aws" / "deployment_info.json"


def print_step(message: str) -> None:
    print(f"\n==> {message}")


def get_default_vpc_and_subnets() -> tuple[str, list[str]]:
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("No default VPC found in this lab account.")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    subnet_ids = [subnet["SubnetId"] for subnet in sorted(subnets, key=lambda item: item["AvailabilityZone"])]
    return vpc_id, subnet_ids


def get_latest_amazon_linux_2023_ami() -> str:
    parameter = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return parameter["Parameter"]["Value"]


def ensure_bucket() -> str:
    print_step(f"Ensuring bucket s3://{S3_BUCKET}")
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise

    s3.put_public_access_block(
        Bucket=S3_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_versioning(
        Bucket=S3_BUCKET,
        VersioningConfiguration={"Status": "Enabled"},
    )
    return S3_BUCKET


def ensure_log_group() -> str:
    log_group_name = f"/{STACK_PREFIX}/api"
    print_step(f"Ensuring CloudWatch log group {log_group_name}")
    try:
        logs.create_log_group(logGroupName=log_group_name)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    logs.put_retention_policy(logGroupName=log_group_name, retentionInDays=14)
    return log_group_name


def ensure_security_group(group_name: str, description: str, vpc_id: str) -> str:
    groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [group_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )["SecurityGroups"]
    if groups:
        return groups[0]["GroupId"]

    response = ec2.create_security_group(
        GroupName=group_name,
        Description=description,
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": [
                    {"Key": "Project", "Value": PROJECT_NAME},
                    {"Key": "Name", "Value": group_name},
                ],
            }
        ],
    )
    return response["GroupId"]


def ensure_security_groups(vpc_id: str) -> tuple[str, str]:
    print_step("Ensuring security groups")
    ec2_sg_id = ensure_security_group(
        f"{STACK_PREFIX}-ec2-sg",
        "DEPORTEData EC2 access",
        vpc_id,
    )
    rds_sg_id = ensure_security_group(
        f"{STACK_PREFIX}-rds-sg",
        "DEPORTEData RDS access",
        vpc_id,
    )

    try:
        ec2.authorize_security_group_ingress(
            GroupId=ec2_sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Web"}],
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 8000,
                    "ToPort": 8000,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "API"}],
                },
            ],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
            raise

    try:
        ec2.authorize_security_group_ingress(
            GroupId=rds_sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 5432,
                    "ToPort": 5432,
                    "UserIdGroupPairs": [{"GroupId": ec2_sg_id}],
                }
            ],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
            raise

    return ec2_sg_id, rds_sg_id


def ensure_db_subnet_group(subnet_ids: list[str]) -> str:
    name = f"{STACK_PREFIX}-db-subnets"
    print_step(f"Ensuring DB subnet group {name}")
    try:
        rds.create_db_subnet_group(
            DBSubnetGroupName=name,
            DBSubnetGroupDescription="DEPORTEData RDS subnet group",
            SubnetIds=subnet_ids,
            Tags=[{"Key": "Project", "Value": PROJECT_NAME}],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"DBSubnetGroupAlreadyExists", "DBSubnetGroupAlreadyExistsFault"}:
            raise
    return name


def ensure_rds(rds_sg_id: str, subnet_group_name: str) -> str:
    print_step(f"Ensuring RDS instance {DB_INSTANCE_ID}")
    try:
        rds.create_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_ID,
            DBInstanceClass="db.t3.micro",
            Engine="postgres",
            EngineVersion=DB_ENGINE_VERSION,
            MasterUsername=DB_USER,
            MasterUserPassword=DB_PASSWORD,
            DBName=DB_NAME,
            AllocatedStorage=20,
            StorageType="gp3",
            PubliclyAccessible=False,
            MultiAZ=False,
            BackupRetentionPeriod=1,
            DeletionProtection=False,
            VpcSecurityGroupIds=[rds_sg_id],
            DBSubnetGroupName=subnet_group_name,
            Tags=[{"Key": "Project", "Value": PROJECT_NAME}],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"DBInstanceAlreadyExists", "DBInstanceAlreadyExistsFault"}:
            raise

    waiter = rds.get_waiter("db_instance_available")
    waiter.wait(
        DBInstanceIdentifier=DB_INSTANCE_ID,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},
    )
    response = rds.describe_db_instances(DBInstanceIdentifier=DB_INSTANCE_ID)
    return response["DBInstances"][0]["Endpoint"]["Address"]


def put_runtime_parameters(db_endpoint: str) -> None:
    print_step("Writing runtime configuration to SSM Parameter Store")
    parameters = {
        f"{PARAMETER_PREFIX}/runtime/S3_BUCKET": (S3_BUCKET, "String"),
        f"{PARAMETER_PREFIX}/runtime/AWS_DEFAULT_REGION": (REGION, "String"),
        f"{PARAMETER_PREFIX}/runtime/DB_HOST": (db_endpoint, "String"),
        f"{PARAMETER_PREFIX}/runtime/DB_NAME": (DB_NAME, "String"),
        f"{PARAMETER_PREFIX}/runtime/DB_USER": (DB_USER, "String"),
        f"{PARAMETER_PREFIX}/runtime/DB_PASSWORD": (DB_PASSWORD, "SecureString"),
        f"{PARAMETER_PREFIX}/runtime/USE_REAL_MODELS": ("true", "String"),
    }
    for name, (value, kind) in parameters.items():
        ssm.put_parameter(Name=name, Value=value, Type=kind, Overwrite=True)


def discover_key_name() -> str:
    if KEY_NAME_OVERRIDE:
        return KEY_NAME_OVERRIDE
    key_pairs = ec2.describe_key_pairs()["KeyPairs"]
    if not key_pairs:
        raise RuntimeError("No EC2 key pair available in this lab account.")
    return key_pairs[0]["KeyName"]


def render_refresh_script() -> str:
    return f"""#!/bin/bash
set -euxo pipefail
APP_DIR="{APP_DIR}"
REGION="{REGION}"
BUCKET="{S3_BUCKET}"
ARCHIVE_KEY="{SOURCE_ARCHIVE_KEY}"
PARAM_PREFIX="{PARAMETER_PREFIX}/runtime"
COMPOSE_BIN="{COMPOSE_BIN}"

mkdir -p "$APP_DIR"
tmp_archive="$(mktemp /tmp/{STACK_PREFIX}.XXXXXX.tar.gz)"
aws s3 cp "s3://$BUCKET/$ARCHIVE_KEY" "$tmp_archive" --region "$REGION"
find "$APP_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +
tar -xzf "$tmp_archive" -C "$APP_DIR"
rm -f "$tmp_archive"

aws ssm get-parameters-by-path \
  --path "$PARAM_PREFIX" \
  --with-decryption \
  --recursive \
  --region "$REGION" \
| jq -r '.Parameters[] | "\\(.Name | split("/")[-1])=\\(.Value)"' > "$APP_DIR/.env"

chown -R ec2-user:ec2-user "$APP_DIR"
cd "$APP_DIR"
"$COMPOSE_BIN" -f "$APP_DIR/docker-compose.yml" down || true
"$COMPOSE_BIN" -f "$APP_DIR/docker-compose.yml" up --build -d
"""


def render_systemd_unit() -> str:
    return f"""[Unit]
Description=DEPORTEData Docker stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={REFRESH_SCRIPT_PATH}
ExecStop={COMPOSE_BIN} -f {APP_DIR}/docker-compose.yml down
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
"""


def build_user_data() -> str:
    return f"""#!/bin/bash
set -euxo pipefail

dnf install -y docker awscli jq tar git curl-minimal
curl -SL {COMPOSE_URL} -o {COMPOSE_BIN}
chmod +x {COMPOSE_BIN}
systemctl enable docker
systemctl start docker

cat >{REFRESH_SCRIPT_PATH} <<'SCRIPT'
{render_refresh_script()}
SCRIPT

chmod +x {REFRESH_SCRIPT_PATH}

cat >{SYSTEMD_SERVICE_PATH} <<'UNIT'
{render_systemd_unit()}
UNIT

systemctl daemon-reload
systemctl enable {SYSTEMD_SERVICE_NAME}
systemctl start {SYSTEMD_SERVICE_NAME}
"""


def find_instance() -> dict | None:
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_NAME]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopped", "stopping"],
            },
        ]
    )
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            return instance
    return None


def wait_for_instance(instance_id: str) -> dict:
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 15, "MaxAttempts": 80})
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]


def terminate_instance(instance_id: str) -> None:
    ec2.terminate_instances(InstanceIds=[instance_id])
    waiter = ec2.get_waiter("instance_terminated")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 15, "MaxAttempts": 80})


def ensure_instance(ec2_sg_id: str, subnet_id: str, ami_id: str) -> dict:
    print_step(f"Ensuring EC2 instance {INSTANCE_NAME}")
    existing = find_instance()
    if existing and REPLACE_INSTANCE:
        print(f"Replacing existing instance {existing['InstanceId']}")
        terminate_instance(existing["InstanceId"])
        existing = None

    if existing:
        state_name = existing["State"]["Name"]
        if state_name in {"stopped", "stopping"}:
            ec2.start_instances(InstanceIds=[existing["InstanceId"]])
        return wait_for_instance(existing["InstanceId"])

    key_name = discover_key_name()
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[ec2_sg_id],
        SubnetId=subnet_id,
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        UserData=build_user_data(),
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": INSTANCE_NAME},
                    {"Key": "Project", "Value": PROJECT_NAME},
                ],
            }
        ],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    return wait_for_instance(instance_id)


def ensure_elastic_ip(instance_id: str) -> str | None:
    if not ENABLE_EIP:
        return None
    print_step(f"Ensuring Elastic IP for {instance_id}")
    addresses = ec2.describe_addresses(
        Filters=[{"Name": "tag:Project", "Values": [PROJECT_NAME]}]
    )["Addresses"]
    if addresses:
        allocation_id = addresses[0]["AllocationId"]
        public_ip = addresses[0]["PublicIp"]
    else:
        allocation = ec2.allocate_address(Domain="vpc")
        allocation_id = allocation["AllocationId"]
        public_ip = allocation["PublicIp"]
        ec2.create_tags(
            Resources=[allocation_id],
            Tags=[
                {"Key": "Project", "Value": PROJECT_NAME},
                {"Key": "Name", "Value": f"{STACK_PREFIX}-eip"},
            ],
        )

    try:
        ec2.associate_address(
            InstanceId=instance_id,
            AllocationId=allocation_id,
            AllowReassociation=True,
        )
    except ClientError as exc:
        print(f"[WARN] Could not associate Elastic IP: {exc}")
        return None
    return public_ip


def ensure_cloudwatch_alarms(instance_id: str) -> None:
    print_step("Ensuring CloudWatch alarms")
    cloudwatch.put_metric_alarm(
        AlarmName=f"{STACK_PREFIX}-ec2-cpu-high",
        ComparisonOperator="GreaterThanThreshold",
        EvaluationPeriods=2,
        MetricName="CPUUtilization",
        Namespace="AWS/EC2",
        Period=300,
        Statistic="Average",
        Threshold=90.0,
        TreatMissingData="notBreaching",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        ActionsEnabled=False,
    )
    cloudwatch.put_metric_alarm(
        AlarmName=f"{STACK_PREFIX}-ec2-status-check",
        ComparisonOperator="GreaterThanThreshold",
        EvaluationPeriods=2,
        MetricName="StatusCheckFailed_Instance",
        Namespace="AWS/EC2",
        Period=300,
        Statistic="Maximum",
        Threshold=0.0,
        TreatMissingData="notBreaching",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        ActionsEnabled=False,
    )
    cloudwatch.put_metric_alarm(
        AlarmName=f"{STACK_PREFIX}-rds-cpu-high",
        ComparisonOperator="GreaterThanThreshold",
        EvaluationPeriods=2,
        MetricName="CPUUtilization",
        Namespace="AWS/RDS",
        Period=300,
        Statistic="Average",
        Threshold=80.0,
        TreatMissingData="notBreaching",
        Dimensions=[{"Name": "DBInstanceIdentifier", "Value": DB_INSTANCE_ID}],
        ActionsEnabled=False,
    )


def wait_for_ssm_online(instance_id: str, timeout_seconds: int = 600) -> None:
    print_step(f"Waiting for SSM on instance {instance_id}")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )
        items = response["InstanceInformationList"]
        if items and items[0]["PingStatus"] == "Online":
            return
        time.sleep(10)
    raise RuntimeError(f"Instance {instance_id} did not become reachable through SSM in time.")


def run_ssm_commands(instance_id: str, commands: list[str], comment: str, timeout_seconds: int = 3600) -> None:
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
        TimeoutSeconds=timeout_seconds,
        Comment=comment,
    )
    command_id = response["Command"]["CommandId"]
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        time.sleep(10)
        try:
            invocation = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        except ssm.exceptions.InvocationDoesNotExist:
            continue

        status = invocation["Status"]
        if status in {"Pending", "InProgress", "Delayed"}:
            continue

        stdout = invocation.get("StandardOutputContent", "")
        stderr = invocation.get("StandardErrorContent", "")
        if status != "Success":
            raise RuntimeError(
                f"SSM command failed with status {status}.\n"
                f"Stdout tail:\n{stdout[-6000:]}\n\nStderr tail:\n{stderr[-6000:]}"
            )

        if stdout.strip():
            try:
                print(stdout[-6000:])
            except UnicodeEncodeError:
                print(stdout[-6000:].encode("ascii", errors="replace").decode("ascii"))
        return

    raise RuntimeError(f"SSM command {command_id} timed out after {timeout_seconds} seconds.")


def bootstrap_instance_via_ssm(instance_id: str) -> None:
    print_step(f"Bootstrapping application on instance {instance_id} via SSM")
    wait_for_ssm_online(instance_id)
    run_ssm_commands(
        instance_id,
        [
            "set -euxo pipefail",
            "dnf install -y docker awscli jq tar git curl-minimal",
            f"curl -SL {COMPOSE_URL} -o {COMPOSE_BIN}",
            f"chmod +x {COMPOSE_BIN}",
            "systemctl enable docker",
            "systemctl start docker",
            f"cat >{REFRESH_SCRIPT_PATH} <<'SCRIPT'\n{render_refresh_script()}\nSCRIPT",
            f"chmod +x {REFRESH_SCRIPT_PATH}",
            f"cat >{SYSTEMD_SERVICE_PATH} <<'UNIT'\n{render_systemd_unit()}\nUNIT",
            "systemctl daemon-reload",
            f"systemctl enable {SYSTEMD_SERVICE_NAME}",
            f"systemctl restart {SYSTEMD_SERVICE_NAME}",
            f"systemctl status {SYSTEMD_SERVICE_NAME} --no-pager",
            f"{COMPOSE_BIN} -f {APP_DIR}/docker-compose.yml ps || true",
        ],
        comment="Bootstrap DEPORTEData stack over SSM",
    )


def main() -> None:
    print(f"Project: {PROJECT_NAME}")
    print(f"Region:  {REGION}")
    print(f"Bucket:  {S3_BUCKET}")

    ensure_bucket()
    ensure_log_group()

    vpc_id, subnet_ids = get_default_vpc_and_subnets()
    ec2_sg_id, rds_sg_id = ensure_security_groups(vpc_id)
    subnet_group_name = ensure_db_subnet_group(subnet_ids)
    db_endpoint = ensure_rds(rds_sg_id, subnet_group_name)
    put_runtime_parameters(db_endpoint)

    ami_id = get_latest_amazon_linux_2023_ami()
    instance = ensure_instance(ec2_sg_id, subnet_ids[0], ami_id)
    public_ip = ensure_elastic_ip(instance["InstanceId"]) or instance.get("PublicIpAddress")
    bootstrap_instance_via_ssm(instance["InstanceId"])
    ensure_cloudwatch_alarms(instance["InstanceId"])

    deployment = {
        "project_name": PROJECT_NAME,
        "region": REGION,
        "bucket": S3_BUCKET,
        "source_archive_key": SOURCE_ARCHIVE_KEY,
        "instance_id": instance["InstanceId"],
        "public_ip": public_ip,
        "web_url": f"http://{public_ip}" if public_ip else None,
        "api_url": f"http://{public_ip}:8000" if public_ip else None,
        "health_url": f"http://{public_ip}/health" if public_ip else None,
        "db_endpoint": db_endpoint,
        "parameter_prefix": PARAMETER_PREFIX,
    }
    OUTPUT_FILE.write_text(json.dumps(deployment, indent=2), encoding="utf-8")

    print("\nDeployment info:")
    print(json.dumps(deployment, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ClientError, WaiterError, RuntimeError) as exc:
        message = f"[ERROR] {exc}"
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode("ascii", errors="replace").decode("ascii"))
        raise
