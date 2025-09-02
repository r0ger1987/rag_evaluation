# 🚨 Évaluation de Fallback - Version Stable

Si vous continuez à avoir des problèmes avec l'évaluation complète, voici une version simplifiée qui fonctionne de manière plus stable.

## 📝 Code de Fallback pour Jupyter

Copiez-collez ce code dans une nouvelle cellule si l'évaluation principale échoue :

```python
# 🛟 ÉVALUATION SIMPLIFIÉE - VERSION FALLBACK

import pandas as pd
from ragas.metrics import faithfulness, answer_correctness
from ragas import evaluate

print("🛟 Évaluation simplifiée avec métriques stables...")

# Utiliser seulement les métriques les plus stables
stable_metrics = [
    faithfulness,      # Très stable, nécessite seulement LLM
    answer_correctness # Stable, compare answer et reference
]

try:
    # Configuration minimale
    for metric in stable_metrics:
        if hasattr(metric, 'llm'):
            metric.llm = ragas_llm
    
    print("✅ Métriques configurées")
    
    # Évaluation avec dataset réduit si nécessaire
    small_dataset = dataset.select(range(min(5, len(dataset))))  # Tester sur 5 premières questions
    
    results = evaluate(
        dataset=small_dataset,
        metrics=stable_metrics,
        llm=ragas_llm,
        raise_exceptions=False
    )
    
    print("\n✅ ÉVALUATION FALLBACK TERMINÉE !")
    
    # Affichage des résultats
    if hasattr(results, 'scores'):
        print("\n📊 SCORES (5 premières questions) :")
        for metric, score in results.scores.items():
            if score is not None:
                emoji = '🟢' if score >= 0.7 else '🟡' if score >= 0.5 else '🔴'
                print(f"   {emoji} {metric}: {score:.3f}")
    
    print("\n💡 Si cette version fonctionne, le problème vient des embeddings ou des métriques de contexte.")
    
except Exception as e:
    print(f"❌ Même l'évaluation fallback échoue : {e}")
    print("\n🔧 Solutions alternatives :")
    print("1. Vérifiez que votre clé API OpenAI est valide")
    print("2. Essayez avec un autre provider (Gemini, Claude)")
    print("3. Testez avec Ollama en local")
```

## 🔧 Métriques par ordre de stabilité

### ✅ Très stables (nécessitent seulement LLM)
- **Faithfulness** : Détecte les hallucinations
- **Answer Correctness** : Compare avec référence

### 🟡 Modérément stables (nécessitent LLM + embeddings)
- **Answer Relevancy** : Peut avoir des problèmes d'embeddings
- **Context Precision** : Dépend de la qualité des ground_truths
- **Context Recall** : Dépend des embeddings pour les comparaisons

## 🎯 Test Minimal

Si même le fallback ne fonctionne pas, testez cette version ultra-minimale :

```python
# Test ultra-minimal
from ragas.metrics import faithfulness

# Test sur 1 seule question
single_item = {
    'question': [dataset[0]['question']],
    'answer': [dataset[0]['answer']], 
    'contexts': [dataset[0]['contexts']]
}

single_dataset = Dataset.from_dict(single_item)

try:
    result = evaluate(
        dataset=single_dataset,
        metrics=[faithfulness],
        llm=ragas_llm
    )
    print("✅ Test minimal réussi !")
    print(f"Faithfulness: {result.scores['faithfulness']:.3f}")
except Exception as e:
    print(f"❌ Test minimal échoué : {e}")
```

## 🔄 Solutions Alternatives

### Option 1: Changer de Provider
```python
# Essayer avec Gemini (souvent plus stable)
RAGAS_LLM_PROVIDER = 'gemini'
GOOGLE_API_KEY = 'your-key'
```

### Option 2: Ollama en Local
```python
# Plus lent mais très stable
RAGAS_LLM_PROVIDER = 'ollama'
OLLAMA_MODEL = 'llama3.2'
```

### Option 3: Évaluation Manuel
```python
# Calculer les métriques une par une
for i in range(len(dataset)):
    item = dataset[i]
    print(f"\nQuestion {i+1}: {item['question'][:50]}...")
    
    # Test faithfulness sur cet item
    try:
        single_result = evaluate(
            dataset=Dataset.from_dict({k: [v] for k, v in item.items()}),
            metrics=[faithfulness],
            llm=ragas_llm
        )
        print(f"✅ Faithfulness: {single_result.scores['faithfulness']:.3f}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
```