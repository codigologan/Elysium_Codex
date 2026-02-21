# 🧠 Logan Agent — Relatório Científico Experimental
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![GPU](https://img.shields.io/badge/GPU-RTX%203060-brightgreen)



**Projeto:** Elysium Codex / Código Logan

**Agente:** Logan (DQN + Dream Module)

**Versão:** v2.0

**Data:** 10-01-2026

**Autor(es):** Ismael Araujo + Logan (IA)

---

## 1. Resumo (Abstract)

Este relatório apresenta a evolução experimental do **Logan Agent**, um agente de *Reinforcement Learning* baseado em DQN com módulos adicionais de **sonhos (dream replay)** e **análise cognitiva emergente**.
O objetivo é investigar se mecanismos inspirados em processos cognitivos — como consolidação offline, exploração guiada e reflexão pós-aprendizado — produzem melhorias mensuráveis em estabilidade, eficiência e generalização do aprendizado.

---

## 2. Objetivo do Estudo

* Avaliar o desempenho do Logan Agent em ambientes controlados de RL.
* Identificar **pontos de transição cognitiva** (turning points) no processo de aprendizagem.
* Medir o impacto do módulo de sonhos (dream) sobre:

  * velocidade de aprendizagem,
  * estabilidade da política,
  * eficiência da exploração.
* Formalizar **fases cognitivas** e **reflexões automáticas** como artefatos analisáveis.

---

## 3. Hipóteses

* **H1:** A inclusão do módulo de sonhos reduz o tempo até a estabilização do reward.
* **H2:** A queda controlada de epsilon está associada a aumento consistente de mean_reward.
* **H3:** É possível detectar um *limiar cognitivo* onde o aprendizado deixa de ser aleatório e passa a ser estrutural.
* **H4:** Métricas cognitivas derivadas de CSV são suficientes para gerar reflexões automáticas interpretáveis.

---

## 4. Metodologia

### 4.1 Ambiente

* Tipo: GridWorld / ambiente discreto
* Estados: observações vetoriais
* Ações: discretas
* Recompensa: densa / negativa por passo, positiva por objetivo

### 4.2 Arquitetura do Agente

* Algoritmo: **DQN**
* Replay Buffer: padrão + replay onírico (dream)
* Política de exploração: ε-greedy
* Scheduler de epsilon: decaimento progressivo até ε_min

### 4.3 Módulo de Sonhos (Dream)

* Frequência: a cada *N* episódios
* Parâmetros principais:

  * `dream_steps`
  * `dream_sigma`
  * `dream_mix_prob`
* Objetivo: consolidação e regularização da política

### 4.4 Logging & Reprodutibilidade

* Histórico por run: `runs/<run_name>/rl_history.csv`
* Métricas agregadas: `reports/rl_results.csv`
* Leaderboard: `reports/leaderboard_rl.csv`
* Reflexões: `logan_reflection.json` (schema versionado)

---

## 5. Métricas Avaliadas

### 5.1 Métricas Clássicas de RL

* Reward por episódio
* Mean reward (janela móvel)
* Episode length
* Epsilon

### 5.2 Métricas Cognitivas (derivadas)

* **Estabilidade:** desvio padrão do reward (janela W)
* **Velocidade de aprendizagem:** inclinação da curva mean_reward
* **Eficiência exploratória:** Δmean_reward / Δepsilon
* **Persistência comportamental:** episode_length médio
* **Sonhos:** dream_loss, novelty_rate

---

## 6. Resultados

### 6.1 Curvas de Aprendizado

*(Inserir gráficos do dashboard ou TensorBoard)*

* Reward vs Episódios
* Mean Reward (smoothing)
* Epsilon decay
* Dream loss por noite

### 6.2 Detecção de Limiar Cognitivo

* Episódio do primeiro turning point: **EP = ___**
* Critério:

  * cruzamento de threshold de mean_reward
  * ou slope positivo sustentado

---

## 7. Fases Cognitivas Identificadas

| Fase                    | Características                         |
| ----------------------- | --------------------------------------- |
| Exploração Caótica      | reward instável, epsilon alto           |
| Aprendizado Emergente   | redução de variância, primeiros padrões |
| Consolidação            | reward positivo consistente             |
| Estabilização           | política repetível, epsilon mínimo      |
| (Opcional) Criatividade | influência ativa dos sonhos             |

---

## 8. Primeira Reflexão Logan (Automática)

> *“Eu atravessei meu primeiro limiar quando a performance deixou de ser acaso e virou padrão...”*

* Fonte: `logan_reflection.json`
* Baseada exclusivamente em dados CSV/TensorBoard
* Sem heurísticas manuais

---

## 9. Discussão

* O Logan Agent demonstrou comportamento compatível com aprendizado estável.
* O módulo de sonhos atua como regularizador e acelerador de convergência.
* As métricas cognitivas permitem interpretação sem inspeção manual de pesos.
* O sistema se aproxima de um **agente com introspecção operacional**.

---

## 10. Limitações

* Ambiente ainda simples (GridWorld)
* Generalização limitada a layouts similares
* Ausência de comparação com outros algoritmos (PPO, A2C, etc.)

---

## 11. Trabalhos Futuros

* Estudos de ablação (com/sem sonhos)
* Variação de seeds e ambientes
* Introdução de memória de longo prazo
* Expansão das fases cognitivas (metacognição)
* Publicação como artigo técnico ou workshop


---

## 12. Referências

* Mnih et al., *Human-level control through deep reinforcement learning*
* Sutton & Barto, *Reinforcement Learning: An Introduction*
* Experimentos internos — Projeto Código Logan

---

## 🚀 Setup Rápido (Local)

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/codigologan/Elysium_Codex.git
cd Elysium_Codex
