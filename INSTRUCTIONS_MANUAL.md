# INSTRUCTIONS MANUELLES - ÉVALUATION RAG

## 📋 Vue d'ensemble

Ce guide explique comment utiliser le système d'évaluation RAG manuel basé sur les fichiers existants du projet. L'évaluation utilise le fichier CSV de référence et le notebook Jupyter dédié pour évaluer les performances de votre système RAG avec les métriques Ragas.

---

## 🗂️ Fichiers disponibles

### Fichier de données de référence
- **Fichier** : `data/reference/reference_qa_manuel_template.csv`
- **Description** : Template avec 10 questions d'exemple du Projet Néo
- **Contenu** : Questions sur management, technique et projet avec réponses de référence

### Notebook d'évaluation
- **Fichier** : `notebooks/SmartRAG_Tutorial_Manuel.ipynb`
- **Description** : Notebook complet pour l'évaluation manuelle avec Ragas
- **Fonctionnalités** : Configuration, chargement, évaluation et analyse

---

## 📝 Structure du fichier CSV

Le fichier `reference_qa_manuel_template.csv` contient **8 colonnes** pour le POC SharePoint :

### Colonnes d'input (à remplir par les métiers)
- `question_id` : Identifiant unique (QSP001, QSP002...)
- `question` : Question à poser au système RAG
- `reference_answer` : Réponse de référence attendue
- `sharepoint_document` : **Nom exact du document SharePoint de référence**

### Colonnes pour l'évaluation Ragas
- `ragas_question` : Question formatée pour Ragas (copie de `question`)
- `ragas_answer` : **Réponse générée par votre système RAG** (à compléter)
- `ragas_contexts` : **Contextes récupérés par votre système** (à compléter)
- `ragas_ground_truth` : **Nom du document SharePoint** (copie de `sharepoint_document`)

### Exemple de données
```csv
question_id,question,reference_answer,sharepoint_document,ragas_question,ragas_answer,ragas_contexts,ragas_ground_truth
QSP001,"Quelle est la procédure pour demander un congé ?","La demande de congé doit être soumise via l'application RH au moins 15 jours avant...","RH_Procedures_Conges_2025.docx","Quelle est la procédure pour demander un congé ?","","","RH_Procedures_Conges_2025.docx"
```

### ⚠️ Important pour le POC
- **`sharepoint_document`** : Les métiers doivent indiquer le nom exact du fichier SharePoint contenant la réponse
- **`ragas_ground_truth`** : Utilisé par Ragas pour vérifier que le bon document a été récupéré

---

## ⚙️ Utilisation du notebook d'évaluation

### Étape 1 : Lancement du notebook
```bash
# Démarrer Jupyter
cd /home/roger/RAG/ragas
jupyter notebook notebooks/SmartRAG_Tutorial_Manuel.ipynb

# Ou avec Jupyter Lab  
jupyter lab notebooks/SmartRAG_Tutorial_Manuel.ipynb
```

### Étape 2 : Configuration des modèles
Le notebook supporte **4 providers LLM** ==> A adapter avec les modèles FM Aws Bedrock

#### OpenAI (recommandé)
```python
RAGAS_LLM_PROVIDER = "openai"
RAGAS_MODEL_NAME = "gpt-4.1-mini"        
OPENAI_API_KEY = "your_openai_key_here"
```

#### Google Gemini
```python
RAGAS_LLM_PROVIDER = "gemini"
RAGAS_MODEL_NAME = "gemini-2.5-flash"    # Avec capacités de raisonnement
GOOGLE_API_KEY = "your_google_key_here"
```

#### Anthropic Claude
```python
RAGAS_LLM_PROVIDER = "claude"
RAGAS_MODEL_NAME = "claude-3-5-haiku-20241022"  # Plus rapide que Claude 3 Opus
ANTHROPIC_API_KEY = "your_anthropic_key_here"
```

#### Ollama (local)
```python
RAGAS_LLM_PROVIDER = "ollama"
RAGAS_MODEL_NAME = "llama3.2"            # Ou mistral, qwen3, etc.
OLLAMA_BASE_URL = "http://localhost:11434"
```

### Étape 3 : Chargement des données
Le notebook charge automatiquement `reference_qa_manuel_template.csv` et valide :
- ✅ **10 questions d'exemple** du POC SharePoint (RH, IT, Sécurité)
- ✅ **Documents SharePoint** : Noms de fichiers .docx et .pdf
- ✅ **Données Ragas** : question, answer (vide), contexts (vide), ground_truth (document SharePoint)

---

## 📊 Les 5 Métriques Ragas Utilisées (Documentation Officielle)

### 1. 🎯 Faithfulness (Fidélité) - Score 0-1
**Définition** : Mesure si la réponse est **factuellement cohérente** avec les contextes récupérés
**Formule** : `Nombre d'affirmations supportées / Total des affirmations dans la réponse`
**Données utilisées** : `ragas_contexts` + `ragas_answer`
**Objectif** : Détecter les hallucinations (informations inventées)

**Exemple** :
- Question : "Qui est le manager de Sophie Martin ?"
- Contexte : "Marc Dubois manage Sophie Martin"
- Réponse : "Marc Dubois est le manager. Il travaille depuis 10 ans."
- Score : 0.5 (1 affirmation vérifiable sur 2 - "10 ans" est inventé)

### 2. ✅ Answer Correctness (Correction) - Score 0-1
**Définition** : Évalue si la réponse est **correcte par rapport à la référence métier**
**Formule** : `α × Similarité_Sémantique + (1-α) × Score_Factuel`
**Données utilisées** : `ragas_answer` + `reference_answer`
**Objectif** : Valider que la réponse correspond aux attentes métier documentées

**Exemple** :
- Référence : "Le délai est de 48-72 heures"
- Réponse RAG : "2 à 3 jours" → Score ~0.85 ✅
- Réponse RAG : "1 semaine" → Score ~0.3 ❌

### 3. 💬 Answer Relevancy (Pertinence) - Score 0-1  
**Définition** : Évalue si la réponse **répond directement à la question** (sans dévier)
**⚠️ DIFFÉRENT** de Answer Correctness : vérifie la pertinence, pas la correction
**Formule** : `Similarité entre question originale et questions générées depuis la réponse`
**Données utilisées** : `ragas_question` + `ragas_answer`

### 4. 🎯 Context Precision (Précision) - Score 0-1
**Définition** : Vérifie si les **contextes pertinents sont bien classés** en tête de liste
**Formule** : `Moyenne(Precision@k)` pour chaque contexte pertinent
**Données utilisées** : `ragas_question` + `ragas_contexts` + `ragas_ground_truth`
**Objectif** : Évaluer la qualité du ranking des documents

### 5. 📚 Context Recall (Rappel) - Score 0-1
**Définition** : Mesure si **tous les contextes importants** ont été récupérés
**Formule** : `Infos de référence trouvées / Total des infos de référence`
**Données utilisées** : `ragas_contexts` + `ragas_ground_truth`
**Pour SharePoint** : Vérifie que les bons documents SharePoint sont dans les contextes récupérés

---

## 🚀 Workflow d'évaluation

### 1. Préparation
- **Les métiers** remplissent les colonnes d'input :
  - `question_id`, `question`, `reference_answer`, `sharepoint_document`
- **Vous** complétez les colonnes Ragas :
  - `ragas_answer` : Réponses générées par **votre système RAG**
  - `ragas_contexts` : Contextes récupérés par **votre système RAG** (incluant noms de fichiers SharePoint)

### 2. Évaluation
Le notebook exécute automatiquement :
```python
# Métriques évaluées
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Évaluation Ragas
results = evaluate(dataset, metrics=metrics, llm=llm_model)
```

### 3. Analyse
Le notebook génère :
- **Statistiques globales** par métrique
- **Analyse par type de document** (.docx, .pdf)
- **Documents SharePoint manqués** (Context Recall faible)
- **Questions problématiques** avec scores faibles
- **Visualisations** (graphiques, heatmaps)

### 4. Export
Sauvegarde automatique :
- **CSV détaillé** : `manual_evaluation_results_detailed.csv`
- **JSON complet** : `manual_evaluation_results.json`
- **Graphiques** : fichiers PNG dans `/results/`

---

## 📈 Interprétation des résultats

### Scores cibles recommandés
| Métrique | Excellent | Bon | À améliorer | Critique |
|----------|-----------|-----|-------------|----------|
| Faithfulness | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
| Answer Relevancy | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Context Precision | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Context Recall | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |

### Actions d'amélioration pour SharePoint
| Score faible | Cause probable | Action recommandée |
|--------------|----------------|-------------------|
| Faithfulness | Hallucinations | Améliorer le prompt, restreindre la génération |
| Answer Relevancy | Réponse hors-sujet | Optimiser la compréhension de la question |
| Context Precision | Mauvais ranking | Améliorer le reranking des documents SharePoint |
| Context Recall | Document SharePoint manqué | Vérifier indexation SharePoint, ajuster les métadonnées |

---

## ⚠️ Bonnes pratiques

### Qualité des données pour SharePoint
- ✅ Vérifier que `sharepoint_document` correspond au nom exact du fichier sur SharePoint
- ✅ S'assurer que `ragas_contexts` contient les noms de documents SharePoint récupérés
- ✅ Valider que `ragas_ground_truth` correspond bien au document SharePoint de référence

### Configuration stable
- ✅ Fixer les seeds pour la reproductibilité
- ✅ Utiliser les mêmes versions de dépendances
- ✅ Documenter la configuration utilisée

### Validation manuelle
- ✅ Examiner quelques réponses manuellement
- ✅ Vérifier la cohérence des scores Ragas
- ✅ Comparer avec d'autres métriques si disponibles