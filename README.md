# Continuous-Time-Quantum-Walks

Algoritmo que calcula a probabilidade de encontrar uma partícula quântica em cada vértice de um grafo utilizando Quantum Walks em tempo contínuo. O código implementa a evolução temporal do estado quântico segundo o Hamiltoniano associado à matriz laplaciana (ou de adjacência) do grafo, permitindo simular a dinâmica e visualizar a distribuição de probabilidades ao longo do tempo.

O algoritmo utiliza o NVIDIA HPC SDK para executar os cálculos de forma otimizada em GPU com as bibliotecas cuSolver e cuBLAS para operações lineares e resolução de autovalores e autovetores com alto desempenho.
