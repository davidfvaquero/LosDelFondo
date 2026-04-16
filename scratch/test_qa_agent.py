import sys
import os

# Add the root directory to sys.path so we can import 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_qa_agent import DataQAAgent

def test_agent():
    print("Inicializando QAAgent para tests...")
    agent = DataQAAgent()
    query = "¿Cuántos registros aproximadamente hay en los datos originados de federado_01.csv?"
    print(f"\nQuerying: {query}")
    res = agent.ask(query)
    print(f"\nRespuesta:\n{res}")

if __name__ == "__main__":
    test_agent()
