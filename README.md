# SmartRAG Evaluation Framework with Ragas

Un framework complet pour évaluer les performances de SmartRAG en utilisant les métriques Ragas via l'intégration Langfuse.

## 📁 Structure du Projet

```
ragas/
├── src/                      # Code source
│   ├── evaluation/          # Modules d'évaluation
│   │   └── smartrag_evaluator.py
│   ├── utils/              # Utilitaires
│   └── config/             # Configuration
├── data/                    # Données
│   ├── reference/          # Questions/réponses de référence
│   │   └── reference_qa.csv
│   └── results/            # Résultats d'évaluation
├── notebooks/              # Jupyter notebooks
├── config/                 # Fichiers de configuration
├── docs/                   # Documentation
├── tests/                  # Tests unitaires
├── evaluate.py             # Script principal
├── .env                    # Variables d'environnement (à créer)
├── .env.template           # Template des variables
└── requirements.txt        # Dépendances Python
```

## 🚀 Installation

### 1. Cloner le projet
```bash
cd /home/roger/RAG/ragas
```

### 2. Créer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration
```bash
# Copier le template de configuration
cp .env.template .env

# Éditer .env avec vos clés API
nano .env
```

## ⚙️ Configuration

### Variables d'environnement essentielles

#### Langfuse (Obligatoire)
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

#### Provider LLM pour Ragas
Choisir un provider et configurer les clés correspondantes :

**OpenAI** (Recommandé)
```bash
RAGAS_LLM_PROVIDER=openai
RAGAS_MODEL_NAME=gpt-4.1-mini
OPENAI_API_KEY=sk-xxxxx
```

**Google Gemini**
```bash
RAGAS_LLM_PROVIDER=gemini
RAGAS_MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=AIzaSyxxxxx
```

**Anthropic Claude**
```bash
RAGAS_LLM_PROVIDER=claude
RAGAS_MODEL_NAME=claude-3-5-haiku-20241022
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

**Ollama (Local)**
```bash
RAGAS_LLM_PROVIDER=ollama
RAGAS_MODEL_NAME=llama2
RAGAS_OLLAMA_BASE_URL=http://localhost:11434
```

## 📊 Préparation des Données

### Format du CSV de référence
Créer un fichier `data/reference/reference_qa.csv` avec les colonnes suivantes :

```csv
question_id,question,reference_answer,expected_contexts,category,difficulty
Q001,"Question de test ?","Réponse attendue","Context1|||Context2|||Context3","Catégorie","Facile"
```

## 🎯 Utilisation

### Évaluation simple
```bash
source venv/bin/activate
python evaluate.py
```

### Avec options personnalisées
```bash
# Regarder les traces des 30 derniers jours
EVALUATION_TIMERANGE=30 python evaluate.py

# Utiliser un modèle différent
RAGAS_MODEL_NAME=gpt-4 python evaluate.py
```

## 📈 Métriques Ragas

Le framework évalue les métriques suivantes :

- **Faithfulness** : La réponse est-elle fidèle aux contextes récupérés ?
- **Answer Relevancy** : La réponse est-elle pertinente par rapport à la question ?
- **Context Precision** : Les contextes récupérés sont-ils pertinents ?
- **Context Recall** : Les contextes contiennent-ils toutes les informations nécessaires ?

## 📝 Résultats

Les résultats sont exportés dans :
- `data/results/smartrag_evaluation_results.csv` : Résultats tabulaires
- `data/results/smartrag_evaluation_results_detailed.json` : Analyse détaillée avec métadonnées

## 🔍 Workflow d'Évaluation

1. **Récupération des traces** : Connexion à Langfuse pour récupérer les traces SmartRAG
2. **Correspondance** : Matching des traces avec les questions de référence
3. **Évaluation Ragas** : Calcul des métriques de qualité
4. **Export** : Sauvegarde des résultats en CSV et JSON

## 🐛 Troubleshooting

### Problème : "No module named 'pandas'"
```bash
pip install pandas langfuse ragas
```

### Problème : Scores à 0.0 avec Gemini/Claude
- Vérifier que les contextes SmartRAG sont pertinents
- S'assurer que les documents sont bien indexés dans SmartRAG
- Essayer avec OpenAI qui est plus tolérant

### Problème : Timeout pendant l'évaluation
Augmenter le timeout dans `.env` :
```bash
EVALUATION_BATCH_SIZE=2  # Réduire la taille du batch
```

## 🤝 Support des Modèles

| Provider | Modèles Supportés | Statut |
|----------|------------------|---------|
| OpenAI | gpt-4.1-mini, gpt-4o-mini, gpt-4, gpt-4-turbo | ✅ Stable |
| Gemini | gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro | ✅ Stable |
| Claude | claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022, claude-3-5-opus | ✅ Stable |
| Ollama | mistral, qwen3, deepseek-r1 | ⚠️ Expérimental |

## 📚 Documentation Additionnelle

- [Configuration Langfuse](https://langfuse.com/docs)
- [Métriques Ragas](https://docs.ragas.io/en/latest/concepts/metrics/)
- [SmartRAG Documentation](à venir)

## 🔄 Mise à jour

Pour mettre à jour les dépendances :
```bash
pip install --upgrade langfuse ragas pandas
```

## 📄 License

Ce projet est sous licence MIT.