#!/usr/bin/env python3
"""
aws/setup_infrastructure.py
============================
Script de aprovisionamiento completo de la infraestructura AWS para DEPORTEData.

Crea:
  - S3 Bucket (modelos + parquets)
  - RDS PostgreSQL (logs de telemetría del chatbot)
  - Security Groups
  - EC2 (backend: API + modelos IA)
  - CloudWatch Log Group + Alarms
  - (Opcional) EC2 frontend si se quiere separado

Uso:
    python aws/setup_infrastructure.py

Variables de entorno requeridas:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION
"""

import boto3
import json
import os

# ─── Configuración ──────────────────────────────────────────────────────────
REGION          = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROJECT_NAME    = "deportedata"
S3_BUCKET       = f"{PROJECT_NAME}-models-data"
DB_NAME         = "deportedata"
DB_USER         = "deporteadmin"
DB_PASSWORD     = "DeporteData2026!"   # Cambiar en producción real
DB_INSTANCE_ID  = f"{PROJECT_NAME}-rds"
EC2_KEY_NAME    = "deportedata-key"    # Nombre del par de claves en AWS
AMI_ID          = "ami-0c02fb55956c7d316"  # Amazon Linux 2023 (us-east-1)
INSTANCE_TYPE   = "t3.large"           # 2 vCPU, 8 GB RAM (suficiente para modelos 0.5B CPU)
DOCKER_HUB_USER  = os.environ.get("DOCKERHUB_USERNAME", "nabreue01")
DOCKER_HUB_TOKEN = os.environ.get("DOCKERHUB_TOKEN", "")

# Clientes boto3
session = boto3.Session(region_name=REGION)
s3      = session.client("s3")
ec2     = session.client("ec2")
rds     = session.client("rds")
logs    = session.client("logs")
cw      = session.client("cloudwatch")
iam     = session.client("iam")


# ─── 1. S3 Bucket ────────────────────────────────────────────────────────────
def create_s3_bucket():
    print(f"\n[1/6] Creando bucket S3: {S3_BUCKET}...")
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        # Bloquear acceso público
        s3.put_public_access_block(
            Bucket=S3_BUCKET,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }
        )
        print(f"  ✅ Bucket s3://{S3_BUCKET} creado.")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  ✅ Bucket s3://{S3_BUCKET} ya existe.")
    return S3_BUCKET


# ─── 2. Subir archivos Parquet a S3 ─────────────────────────────────────────
def upload_parquets():
    print("\n[2/6] Subiendo archivos .parquet a S3...")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = {
        "data/processed/federados.parquet": "data/processed/federados.parquet",
        "data/processed/gasto.parquet":     "data/processed/gasto.parquet",
    }
    for local_rel, s3_key in files.items():
        local_path = os.path.join(base, local_rel)
        if os.path.exists(local_path):
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"  ✅ Subido: {local_rel} → s3://{S3_BUCKET}/{s3_key}")
        else:
            print(f"  ⚠️  No encontrado: {local_path}")


# ─── 3. CloudWatch Log Group ─────────────────────────────────────────────────
def create_cloudwatch():
    print("\n[3/6] Configurando CloudWatch...")
    log_group = "/deportedata/api"
    try:
        logs.create_log_group(logGroupName=log_group)
        logs.put_retention_policy(logGroupName=log_group, retentionInDays=30)
        print(f"  ✅ Log group creado: {log_group} (retención: 30 días)")
    except logs.exceptions.ResourceAlreadyExistsException:
        print(f"  ✅ Log group ya existe: {log_group}")

    # Alarma: API sin responder durante 5 minutos
    cw.put_metric_alarm(
        AlarmName="DEPORTEData-API-HealthCheck",
        ComparisonOperator="LessThanThreshold",
        EvaluationPeriods=2,
        MetricName="HealthyHostCount",
        Namespace="AWS/ApplicationELB",
        Period=300,
        Statistic="Average",
        Threshold=1.0,
        ActionsEnabled=False,
        AlarmDescription="API de DEPORTEData no responde",
        TreatMissingData="breaching",
    )
    print("  ✅ Alarma CloudWatch configurada.")


# ─── 4. RDS PostgreSQL ───────────────────────────────────────────────────────
def create_rds(sg_id: str, subnet_ids: list[str]) -> str:
    print(f"\n[4/6] Creando RDS PostgreSQL: {DB_INSTANCE_ID}...")
    try:
        rds.create_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_ID,
            DBInstanceClass="db.t3.micro",
            Engine="postgres",
            EngineVersion="15.4",
            MasterUsername=DB_USER,
            MasterUserPassword=DB_PASSWORD,
            DBName=DB_NAME,
            AllocatedStorage=20,
            StorageType="gp2",
            PubliclyAccessible=False,
            VpcSecurityGroupIds=[sg_id],
            BackupRetentionPeriod=7,
            DeletionProtection=False,
            Tags=[{"Key": "Project", "Value": "DEPORTEData"}],
        )
        print("  ⏳ RDS creándose... (puede tardar ~5 minutos)")

        # Esperar a que esté disponible
        waiter = rds.get_waiter("db_instance_available")
        waiter.wait(DBInstanceIdentifier=DB_INSTANCE_ID,
                    WaiterConfig={"Delay": 30, "MaxAttempts": 30})

        resp = rds.describe_db_instances(DBInstanceIdentifier=DB_INSTANCE_ID)
        endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
        print(f"  ✅ RDS disponible en: {endpoint}")
        return endpoint

    except rds.exceptions.DBInstanceAlreadyExistsFault:
        resp = rds.describe_db_instances(DBInstanceIdentifier=DB_INSTANCE_ID)
        endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
        print(f"  ✅ RDS ya existe: {endpoint}")
        return endpoint


# ─── 5. Security Groups ──────────────────────────────────────────────────────
def create_security_groups(vpc_id: str) -> tuple[str, str]:
    print(f"\n[5/6] Creando Security Groups en VPC {vpc_id}...")

    # SG para EC2 (API + Dashboard)
    try:
        sg_ec2 = ec2.create_security_group(
            GroupName="deportedata-ec2-sg",
            Description="DEPORTEData EC2 - API y Dashboard",
            VpcId=vpc_id,
        )["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_ec2,
            IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": 22,   "ToPort": 22,   "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 80,   "ToPort": 80,   "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 443,  "ToPort": 443,  "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ]
        )
        print(f"  ✅ SG EC2: {sg_ec2}")
    except ec2.exceptions.ClientError as e:
        if "already exists" in str(e):
            sg_ec2 = [sg["GroupId"] for sg in ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": ["deportedata-ec2-sg"]}])["SecurityGroups"]][0]
            print(f"  ✅ SG EC2 ya existe: {sg_ec2}")
        else:
            raise

    # SG para RDS (solo acceso desde EC2)
    try:
        sg_rds = ec2.create_security_group(
            GroupName="deportedata-rds-sg",
            Description="DEPORTEData RDS - solo acceso desde EC2",
            VpcId=vpc_id,
        )["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_rds,
            IpPermissions=[{
                "IpProtocol": "tcp",
                "FromPort": 5432,
                "ToPort": 5432,
                "UserIdGroupPairs": [{"GroupId": sg_ec2}],
            }]
        )
        print(f"  ✅ SG RDS: {sg_rds}")
    except ec2.exceptions.ClientError as e:
        if "already exists" in str(e):
            sg_rds = [sg["GroupId"] for sg in ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": ["deportedata-rds-sg"]}])["SecurityGroups"]][0]
            print(f"  ✅ SG RDS ya existe: {sg_rds}")
        else:
            raise

    return sg_ec2, sg_rds


# ─── 6. EC2 Instance (API + Dashboard en la misma instancia) ─────────────────
def create_ec2(sg_id: str, db_endpoint: str) -> str:
    print(f"\n[6/6] Lanzando EC2 ({INSTANCE_TYPE})...")

    # User data: instala Docker, descarga imágenes y arranca
    user_data = f"""#!/bin/bash
set -e
exec > /var/log/deportedata-setup.log 2>&1

echo "=== DEPORTEData EC2 Setup ==="
dnf update -y
dnf install -y docker git

# Instalar Docker Compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
     -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

systemctl enable docker
systemctl start docker

# Login a Docker Hub
echo "{DOCKER_HUB_TOKEN}" | docker login -u {DOCKER_HUB_USER} --password-stdin

# Clone repository (branch dev)
mkdir -p /home/ec2-user/app
git clone -b deploy/aws-infrastructure https://github.com/davidfvaquero/LosDelFondo.git /home/ec2-user/app
chown -R ec2-user:ec2-user /home/ec2-user/app

# Crear .env con variables del entorno
cat > /home/ec2-user/app/.env << 'ENVEOF'
S3_BUCKET={S3_BUCKET}
AWS_DEFAULT_REGION={REGION}
AWS_ACCESS_KEY_ID={os.environ.get("AWS_ACCESS_KEY_ID", "")}
AWS_SECRET_ACCESS_KEY={os.environ.get("AWS_SECRET_ACCESS_KEY", "")}
AWS_SESSION_TOKEN={os.environ.get("AWS_SESSION_TOKEN", "")}
DB_HOST={db_endpoint}
DB_NAME={DB_NAME}
DB_USER={DB_USER}
DB_PASSWORD={DB_PASSWORD}
ENVEOF

# Arrancar servicios (build nativo en EC2)
cd /home/ec2-user/app
docker compose build
docker compose up -d

# Configurar reinicio automático al boot
cat > /etc/systemd/system/deportedata.service << 'SVCEOF'
[Unit]
Description=DEPORTEData Application
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/app
EnvironmentFile=/home/ec2-user/app/.env
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose up -d
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl enable deportedata.service
echo "=== Setup completado ==="
"""

    try:
        resp = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            KeyName=EC2_KEY_NAME,
            SecurityGroupIds=[sg_id],
            UserData=user_data,
            IamInstanceProfile={"Name": "LabInstanceProfile"},  # Rol de AWS Academy
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 40, "VolumeType": "gp3"},  # 40 GB para modelos
            }],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name",    "Value": "deportedata-backend"},
                    {"Key": "Project", "Value": "DEPORTEData"},
                ]
            }],
        )

        instance_id = resp["Instances"][0]["InstanceId"]
        print(f"  ⏳ EC2 lanzada: {instance_id} — esperando que esté running...")

        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])

        desc = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "N/A")
        print("  ✅ EC2 corriendo!")
        print(f"  📍 IP Pública: {public_ip}")
        print(f"  🌐 API:        http://{public_ip}:8000")
        print(f"  🌐 Dashboard:  http://{public_ip}:8501")
        return public_ip

    except Exception as e:
        print(f"  ❌ Error lanzando EC2: {e}")
        raise


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DEPORTEData — Aprovisionamiento AWS")
    print(f"  Región: {REGION}")
    print("=" * 60)

    # Obtener VPC y subnets por defecto
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    subnet_ids = [s["SubnetId"] for s in subnets]
    print(f"\n  VPC por defecto: {vpc_id}")

    # Ejecutar pasos
    create_s3_bucket()
    upload_parquets()
    create_cloudwatch()
    sg_ec2, sg_rds = create_security_groups(vpc_id)
    db_endpoint = create_rds(sg_rds, subnet_ids)
    public_ip   = create_ec2(sg_ec2, db_endpoint)

    # Resumen final
    print("\n" + "=" * 60)
    print("  ✅ INFRAESTRUCTURA DESPLEGADA")
    print("=" * 60)
    print(f"  S3 Bucket:   s3://{S3_BUCKET}")
    print(f"  RDS:         {db_endpoint}:5432 / db={DB_NAME}")
    print(f"  EC2 API:     http://{public_ip}:8000")
    print(f"  EC2 Dash:    http://{public_ip}:8501")
    print("  CloudWatch:  /deportedata/api")
    print("\n  ⚠️  Los modelos de IA se descargarán desde HuggingFace/S3")
    print("  ⚠️  La API puede tardar 2-5 min en estar lista (primera carga)")
    print("=" * 60)

    # Guardar IPs en archivo para referencia
    with open("aws/deployment_info.json", "w") as f:
        json.dump({
            "public_ip": public_ip,
            "api_url": f"http://{public_ip}:8000",
            "dashboard_url": f"http://{public_ip}:8501",
            "s3_bucket": S3_BUCKET,
            "rds_endpoint": db_endpoint,
            "region": REGION,
        }, f, indent=2)
    print("\n  📄 Info guardada en aws/deployment_info.json")


if __name__ == "__main__":
    main()
