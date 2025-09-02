#!/usr/bin/env python3
"""
Script d'évaluation SmartRAG + Langfuse avec Ragas
Adapté à l'API Langfuse 3.x
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Import Langfuse et Ragas
from langfuse import Langfuse
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def load_config():
    """Charge la configuration depuis le fichier .env"""
    load_dotenv()
    
    config = {
        'langfuse_public_key': os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        'langfuse_secret_key': os.getenv("LANGFUSE_SECRET_KEY", ""),
        'langfuse_base_url': os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        'reference_csv': os.getenv("REFERENCE_CSV", "./data/reference/reference_qa.csv"),
        'output_csv': os.getenv("OUTPUT_CSV", "./data/results/smartrag_evaluation_results.csv"),
        'output_json': os.getenv("OUTPUT_JSON", "./data/results/smartrag_evaluation_results_detailed.json"),
        'smartrag_project_name': os.getenv("SMARTRAG_PROJECT_NAME", ""),
        'evaluation_timerange': int(os.getenv("EVALUATION_TIMERANGE", "7")),
        'min_confidence_score': float(os.getenv("MIN_CONFIDENCE_SCORE", "0.0")),
        'include_failed_traces': os.getenv("INCLUDE_FAILED_TRACES", "false").lower() == "true",
        
        # Configuration LLM pour Ragas
        'openai_api_key': os.getenv("OPENAI_API_KEY", ""),
        'ragas_llm_provider': os.getenv("RAGAS_LLM_PROVIDER", "openai"),
        'ragas_model_name': os.getenv("RAGAS_MODEL_NAME", "gpt-3.5-turbo"),
        'google_api_key': os.getenv("GOOGLE_API_KEY", ""),
        'anthropic_api_key': os.getenv("ANTHROPIC_API_KEY", ""),
        'ragas_ollama_base_url': os.getenv("RAGAS_OLLAMA_BASE_URL", "http://localhost:11434"),
        'ragas_ollama_model': os.getenv("RAGAS_OLLAMA_MODEL", "llama2")
    }
    
    if not config['langfuse_public_key'] or not config['langfuse_secret_key']:
        raise ValueError("LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY sont requis")
    
    return config

def connect_langfuse(config):
    """Connexion à Langfuse"""
    return Langfuse(
        public_key=config['langfuse_public_key'],
        secret_key=config['langfuse_secret_key'],
        host=config['langfuse_base_url']
    )

def load_reference_data(csv_path):
    """Charge les données de référence depuis le CSV"""
    try:
        df_reference = pd.read_csv(csv_path, encoding='utf-8')
        print(f"Jeu de référence chargé : {len(df_reference)} questions")
        
        # Validation des colonnes obligatoires
        required_cols = ['question_id', 'question', 'reference_answer']
        missing_cols = [col for col in required_cols if col not in df_reference.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing_cols}")
        
        # Nettoyage des données
        df_reference['question'] = df_reference['question'].fillna('').astype(str).str.strip()
        df_reference['reference_answer'] = df_reference['reference_answer'].fillna('').astype(str).str.strip()
        
        # Traitement des contextes attendus
        if 'expected_contexts' in df_reference.columns:
            df_reference['expected_contexts_list'] = df_reference['expected_contexts'].fillna('').apply(
                lambda x: x.split('|||') if x else []
            )
        else:
            df_reference['expected_contexts_list'] = [[] for _ in range(len(df_reference))]
        
        return df_reference
        
    except FileNotFoundError:
        print(f"ERREUR : Fichier {csv_path} non trouvé.")
        print("Utilisation du template par défaut...")
        return create_default_reference_data()

def create_default_reference_data():
    """Crée un jeu de données de référence par défaut pour test"""
    data = {
        'question_id': ['Q001', 'Q002', 'Q003'],
        'question': [
            'Quelle est la date limite du premier livrable de Sophie Martin ?',
            'Qui est le manager de Marc Dubois ?',
            'Quel framework est utilisé pour le Projet Néo ?'
        ],
        'reference_answer': [
            'Le premier PoC de Sophie Martin doit être livré pour le 1er mars 2025.',
            'Le manager de Marc Dubois est Carole Lambert.',
            'Le framework utilisé est TensorFlow.'
        ],
        'expected_contexts': [
            'Compte-rendu de réunion|||PoC delivery',
            'Équipe projet|||Management hierarchy',
            'Technologies|||Framework choice'
        ],
        'category': ['Livrable', 'Management', 'Technique'],
        'difficulty': ['Facile', 'Facile', 'Facile']
    }
    
    df = pd.DataFrame(data)
    df['expected_contexts_list'] = df['expected_contexts'].apply(lambda x: x.split('|||'))
    return df

def fetch_smartrag_traces(langfuse_client, config):
    """Récupère les traces SmartRAG depuis Langfuse"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config['evaluation_timerange'])
    
    print(f"Récupération des traces du {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}")
    
    traces_data = []
    page = 1
    limit = 50
    
    while True:
        try:
            # Utilisation de la nouvelle API Langfuse 3.x
            traces_response = langfuse_client.api.trace.list(
                page=page,
                limit=limit,
                from_timestamp=start_date,
                to_timestamp=end_date
            )
            
            if not traces_response.data:
                break
                
            print(f"Page {page} : {len(traces_response.data)} traces récupérées")
            
            for trace in traces_response.data:
                # Filtrage par nom de projet si spécifié
                if config['smartrag_project_name'] and config['smartrag_project_name'] not in str(trace.name):
                    continue
                
                # Extraction des informations de la trace
                trace_data = {
                    'trace_id': trace.id,
                    'timestamp': trace.timestamp,
                    'name': trace.name,
                    'input': trace.input,
                    'output': trace.output,
                    'metadata': trace.metadata or {},
                    'tags': trace.tags or [],
                    'user_id': trace.user_id,
                    'session_id': trace.session_id,
                    'latency': trace.latency,
                    'total_cost': trace.total_cost
                }
                
                traces_data.append(trace_data)
                
            page += 1
            
            # Protection contre les boucles infinies
            if page > 50:  # Max 2500 traces
                print("Limite de pages atteinte (50)")
                break
                
        except Exception as e:
            print(f"Erreur lors de la récupération des traces (page {page}): {e}")
            break
    
    print(f"Total des traces récupérées : {len(traces_data)}")
    return traces_data

def extract_rag_components(trace):
    """Extrait les composants RAG d'une trace"""
    
    # Extraction de la question
    question = ""
    if isinstance(trace.get('input'), dict):
        input_data = trace['input']
        
        # Format SmartRAG avec history
        if 'history' in input_data and isinstance(input_data['history'], list):
            for msg in input_data['history']:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
        
        # Format messages chat standard
        elif 'messages' in input_data and isinstance(input_data['messages'], list):
            for msg in input_data['messages']:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
        
        # Format direct prompt/question
        else:
            question = input_data.get('prompt', input_data.get('question', input_data.get('query', '')))
            # Pour les traces 'chat' qui ont le prompt direct
            if not question and 'prompt' in input_data:
                prompt_text = input_data['prompt']
                if isinstance(prompt_text, str) and 'Quelle est' in prompt_text:
                    # Extraction de la question depuis le prompt
                    lines = prompt_text.split('\n')
                    for line in lines:
                        if line.strip().startswith('Quelle est') or line.strip().startswith('Qui ') or line.strip().startswith('Comment '):
                            question = line.strip()
                            break
    
    elif isinstance(trace.get('input'), str):
        question = trace['input']
    
    # Extraction de la réponse
    answer = ""
    if isinstance(trace.get('output'), dict):
        output_data = trace['output']
        answer = output_data.get('output', output_data.get('answer', output_data.get('response', output_data.get('text', ''))))
        
        # Pour les traces 'chat' avec time_elapsed format
        if not answer and 'time_elapsed:' in output_data:
            # Recherche d'une réponse dans les autres clés
            for key, value in output_data.items():
                if isinstance(value, str) and len(value) > 50 and ('Sophie Martin' in value or 'Marc Dubois' in value):
                    answer = value
                    break
    
    elif isinstance(trace.get('output'), str):
        answer = trace['output']
    
    # Extraction des contextes depuis le knowledge base dans system prompt
    contexts = []
    if isinstance(trace.get('input'), dict):
        input_data = trace['input']
        
        # Recherche dans le système prompt
        if 'system' in input_data:
            system_content = input_data['system']
            if 'knowledge base' in system_content.lower():
                # Extraction des segments ID avec leur contenu
                lines = system_content.split('\n')
                current_context = ""
                for line in lines:
                    if line.startswith('ID:'):
                        if current_context:
                            contexts.append(current_context.strip())
                        current_context = line
                    elif current_context and line.strip():
                        current_context += "\n" + line
                
                # Ajouter le dernier contexte
                if current_context:
                    contexts.append(current_context.strip())
        
        # Recherche dans le prompt direct pour les traces 'chat'
        elif 'prompt' in input_data:
            prompt_content = input_data['prompt']
            if isinstance(prompt_content, str) and 'knowledge base' in prompt_content.lower():
                # Même logique d'extraction
                lines = prompt_content.split('\n')
                current_context = ""
                for line in lines:
                    if line.startswith('ID:'):
                        if current_context:
                            contexts.append(current_context.strip())
                        current_context = line
                    elif current_context and line.strip() and not line.startswith('------'):
                        current_context += "\n" + line
                
                if current_context:
                    contexts.append(current_context.strip())
    
    return {
        'trace_id': trace['trace_id'],
        'timestamp': trace['timestamp'],
        'question': str(question).strip(),
        'answer': str(answer).strip(),
        'contexts': contexts,
        'session_id': trace.get('session_id'),
        'user_id': trace.get('user_id'),
        'metadata': trace.get('metadata', {}),
        'tags': trace.get('tags', []),
        'latency': trace.get('latency', 0),
        'total_cost': trace.get('total_cost', 0),
        'trace_name': trace.get('name', '')
    }

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcule la similarité entre deux textes"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def match_traces_to_reference(df_traces, df_reference, similarity_threshold=0.7):
    """Fait correspondre les traces aux questions de référence"""
    
    matched_data = []
    
    for _, ref_row in df_reference.iterrows():
        ref_question = ref_row['question']
        best_match = None
        best_similarity = 0
        
        # Recherche de la meilleure correspondance en priorisant les traces avec de vraies réponses
        for _, trace_row in df_traces.iterrows():
            similarity = calculate_similarity(ref_question, trace_row['question'])
            
            if similarity >= similarity_threshold:
                # Prioriser les traces avec des réponses qui ne sont pas des métadonnées
                is_better_answer = (
                    trace_row['trace_name'] == 'chat_streamly' or
                    (len(trace_row['answer']) > 0 and 'Time elapsed' not in trace_row['answer'] and 'Query:' not in trace_row['answer'])
                )
                
                # Mettre à jour si c'est une meilleure correspondance
                if similarity > best_similarity or (similarity == best_similarity and is_better_answer):
                    best_similarity = similarity
                    best_match = trace_row
        
        if best_match is not None:
            matched_row = {
                # Données de référence
                'question_id': ref_row['question_id'],
                'reference_question': ref_row['question'],
                'reference_answer': ref_row['reference_answer'],
                'expected_contexts': ref_row.get('expected_contexts_list', []),
                'category': ref_row.get('category', ''),
                'difficulty': ref_row.get('difficulty', ''),
                
                # Données de la trace
                'trace_id': best_match['trace_id'],
                'actual_question': best_match['question'],
                'actual_answer': best_match['answer'],
                'retrieved_contexts': best_match['contexts'],
                'question_similarity': best_similarity,
                
                # Métadonnées de la trace
                'timestamp': best_match['timestamp'],
                'session_id': best_match.get('session_id'),
                'user_id': best_match.get('user_id'),
                'latency': best_match.get('latency', 0),
                'total_cost': best_match.get('total_cost', 0),
                'trace_name': best_match.get('trace_name', ''),
                'trace_metadata': best_match.get('metadata', {}),
                'trace_tags': best_match.get('tags', [])
            }
            matched_data.append(matched_row)
        else:
            print(f"Pas de correspondance pour la question: {ref_row['question_id']} (seuil: {similarity_threshold})")
    
    return pd.DataFrame(matched_data)

def setup_ragas_llm(config):
    """Configure le modèle LLM pour Ragas selon le provider choisi"""
    
    provider = config.get('ragas_llm_provider', 'openai').lower()
    model_name = config.get('ragas_model_name', 'gpt-3.5-turbo')
    
    print(f"Configuration Ragas avec {provider} ({model_name})")
    
    try:
        if provider == 'openai':
            from langchain_openai import ChatOpenAI
            from langchain_openai import OpenAIEmbeddings
            
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=config.get('openai_api_key')
            )
            embeddings = OpenAIEmbeddings(api_key=config.get('openai_api_key'))
            
        elif provider == 'gemini':
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            llm = ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro",
                temperature=0.0,
                google_api_key=config.get('google_api_key')
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.get('google_api_key')
            )
            
        elif provider == 'claude':
            from langchain_anthropic import ChatAnthropic
            # Note: Anthropic n'a pas d'embeddings, utilise OpenAI en fallback
            from langchain_openai import OpenAIEmbeddings
            
            llm = ChatAnthropic(
                model=model_name or "claude-3-sonnet-20240229",
                temperature=0.0,
                anthropic_api_key=config.get('anthropic_api_key')
            )
            embeddings = OpenAIEmbeddings(api_key=config.get('openai_api_key'))
            
        elif provider == 'ollama':
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import OllamaEmbeddings
            
            llm = Ollama(
                model=config.get('ragas_ollama_model', 'llama2'),
                base_url=config.get('ragas_ollama_base_url', 'http://localhost:11434')
            )
            embeddings = OllamaEmbeddings(
                model=config.get('ragas_ollama_model', 'llama2'),
                base_url=config.get('ragas_ollama_base_url', 'http://localhost:11434')
            )
            
        else:
            raise ValueError(f"Provider non supporté: {provider}")
        
        # Configuration globale de Ragas
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        return ragas_llm, ragas_embeddings
        
    except ImportError as e:
        print(f"Erreur d'import pour {provider}: {e}")
        print(f"Installez les dépendances: pip install langchain-{provider}")
        return None, None
    except Exception as e:
        print(f"Erreur de configuration {provider}: {e}")
        return None, None

def evaluate_with_ragas(df_matched, config):
    """Évalue les données matchées avec Ragas"""
    
    if len(df_matched) == 0:
        print("Aucune donnée à évaluer")
        return df_matched
    
    print(f"Évaluation Ragas de {len(df_matched)} questions...")
    
    # Configuration du modèle LLM pour Ragas
    ragas_llm, ragas_embeddings = setup_ragas_llm(config)
    
    if not ragas_llm or not ragas_embeddings:
        print("Impossible de configurer Ragas, évaluation sautée")
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            df_matched[metric] = None
        return df_matched
    
    # Préparation des données pour Ragas
    eval_data = df_matched.copy()
    
    # Nettoyage des contextes
    eval_data['contexts_cleaned'] = eval_data['retrieved_contexts'].apply(
        lambda x: [str(ctx).strip() for ctx in x] if isinstance(x, list) and x else ["Aucun contexte disponible"]
    )
    
    # Mapping des colonnes pour Ragas
    column_map = {
        "question": "reference_question",
        "answer": "actual_answer", 
        "contexts": "contexts_cleaned",
        "ground_truth": "reference_answer"
    }
    
    # Métriques Ragas avec LLM personnalisé
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        # Configuration des métriques avec le LLM personnalisé
        faithfulness.llm = ragas_llm
        faithfulness.embeddings = ragas_embeddings
        
        answer_relevancy.llm = ragas_llm  
        answer_relevancy.embeddings = ragas_embeddings
        
        context_precision.llm = ragas_llm
        context_precision.embeddings = ragas_embeddings
        
        context_recall.llm = ragas_llm
        context_recall.embeddings = ragas_embeddings
        
        metrics = [
            faithfulness,           # Fidélité au contexte
            answer_relevancy,       # Pertinence de la réponse
            context_precision,      # Précision du contexte
            context_recall          # Rappel du contexte
        ]
        
        # Conversion en Dataset Ragas
        from datasets import Dataset
        
        # Préparation des données au format Ragas
        dataset_dict = {
            "question": eval_data["reference_question"].tolist(),
            "answer": eval_data["actual_answer"].tolist(),
            "contexts": eval_data["contexts_cleaned"].tolist(),
            "ground_truth": eval_data["reference_answer"].tolist()
        }
        
        ragas_dataset = Dataset.from_dict(dataset_dict)
        
        # Évaluation Ragas
        ragas_results = evaluate(ragas_dataset, metrics=metrics)
        
        print("\n=== RÉSULTATS RAGAS ===")
        print(f"Modèle utilisé: {config.get('ragas_llm_provider')} ({config.get('ragas_model_name')})")
        print(f"Type de résultats: {type(ragas_results)}")
        print(f"Contenu: {ragas_results}")
        print("\nScores moyens:")
        
        # Extraction des scores depuis EvaluationResult
        if hasattr(ragas_results, '_scores_dict'):
            # Accès aux scores via l'attribut interne _scores_dict
            raw_scores_dict = ragas_results._scores_dict
            scores_dict = {}
            
            for metric, score in raw_scores_dict.items():
                # Gérer les cas où score est une liste ou une valeur unique
                if isinstance(score, list):
                    avg_score = sum(score) / len(score) if score else 0
                    print(f"  {metric}: {avg_score:.4f} (moy. de {len(score)} valeurs)")
                    scores_dict[metric] = avg_score
                else:
                    print(f"  {metric}: {score:.4f}")
                    scores_dict[metric] = score
            
            # Conversion en DataFrame pour compatibilité
            scores_df = pd.DataFrame([scores_dict])
            
        elif str(ragas_results).startswith('{') and str(ragas_results).endswith('}'):
            # Parsing de la représentation string
            import ast
            scores_dict = ast.literal_eval(str(ragas_results))
            
            for metric, score in scores_dict.items():
                print(f"  {metric}: {score:.4f}")
            
            scores_df = pd.DataFrame([scores_dict])
            
        else:
            # Fallback pour autres formats
            scores_dict = {}
            scores_df = pd.DataFrame()
        
        # Fusion des résultats
        df_final = df_matched.copy()
        
        # Ajout des scores Ragas
        for i, (_, row) in enumerate(df_matched.iterrows()):
            for col in scores_df.columns:
                if col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    df_final.at[row.name, col] = scores_df.iloc[min(i, len(scores_df)-1)][col] if len(scores_df) > 0 else None
        
        return df_final
        
    except Exception as e:
        print(f"Erreur lors de l'évaluation Ragas: {e}")
        import traceback
        traceback.print_exc()
        # Ajout de colonnes vides pour les métriques
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            df_matched[metric] = None
        return df_matched

def create_visualizations(df_final):
    """Crée les visualisations des résultats"""
    
    if len(df_final) == 0 or 'faithfulness' not in df_final.columns:
        print("Aucune donnée disponible pour les visualisations")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Graphique 1 : Scores moyens par métrique
    plt.subplot(2, 3, 1)
    metrics_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    
    available_metrics = [col for col in metrics_cols if col in df_final.columns and df_final[col].notna().any()]
    
    if available_metrics:
        means = [df_final[col].mean() for col in available_metrics]
        plt.bar(range(len(available_metrics)), means, color='skyblue')
        plt.xticks(range(len(available_metrics)), available_metrics, rotation=45, ha='right')
        plt.title('Scores Ragas moyens')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for i, v in enumerate(means):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Graphique 2 : Distribution des scores de fidélité
    plt.subplot(2, 3, 2)
    if 'faithfulness' in df_final.columns and df_final['faithfulness'].notna().any():
        plt.hist(df_final['faithfulness'].dropna(), bins=10, color='lightcoral', alpha=0.7)
        plt.title('Distribution - Faithfulness')
        plt.xlabel('Score')
        plt.ylabel('Fréquence')
    
    # Graphique 3 : Corrélation similarité vs. qualité
    plt.subplot(2, 3, 3)
    if 'question_similarity' in df_final.columns and 'answer_relevancy' in df_final.columns:
        valid_data = df_final[df_final['answer_relevancy'].notna() & df_final['question_similarity'].notna()]
        if len(valid_data) > 0:
            plt.scatter(valid_data['question_similarity'], valid_data['answer_relevancy'], alpha=0.6)
            plt.xlabel('Similarité question')
            plt.ylabel('Answer Relevancy')
            plt.title('Similarité vs. Pertinence')
    
    # Graphique 4 : Latence vs. qualité
    plt.subplot(2, 3, 4)
    if 'latency' in df_final.columns and 'faithfulness' in df_final.columns:
        valid_data = df_final[df_final['faithfulness'].notna() & df_final['latency'].notna()]
        if len(valid_data) > 0:
            plt.scatter(valid_data['latency'], valid_data['faithfulness'], alpha=0.6)
            plt.xlabel('Latence (s)')
            plt.ylabel('Faithfulness')
            plt.title('Latence vs. Fidélité')
    
    # Graphique 5 : Scores par catégorie
    plt.subplot(2, 3, 5)
    if 'category' in df_final.columns and df_final['category'].notna().any() and 'faithfulness' in df_final.columns:
        category_scores = df_final.groupby('category')['faithfulness'].mean().sort_values(ascending=False)
        if len(category_scores) > 1:
            plt.bar(range(len(category_scores)), category_scores.values, color='lightgreen')
            plt.xticks(range(len(category_scores)), category_scores.index, rotation=45, ha='right')
            plt.title('Faithfulness par catégorie')
            plt.ylabel('Score moyen')
    
    # Graphique 6 : Évolution temporelle
    plt.subplot(2, 3, 6)
    if 'timestamp' in df_final.columns and len(df_final) > 1:
        df_final['date'] = pd.to_datetime(df_final['timestamp']).dt.date
        daily_scores = df_final.groupby('date')['faithfulness'].mean()
        if len(daily_scores) > 1:
            plt.plot(daily_scores.index, daily_scores.values, marker='o', color='purple')
            plt.title('Évolution Faithfulness')
            plt.xlabel('Date')
            plt.ylabel('Score moyen')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques détaillées
    print("\n=== STATISTIQUES DÉTAILLÉES ===")
    
    for metric in available_metrics:
        values = df_final[metric].dropna()
        if len(values) > 0:
            print(f"\n{metric.upper()}:")
            print(f"  Moyenne: {values.mean():.4f}")
            print(f"  Médiane: {values.median():.4f}")
            print(f"  Écart-type: {values.std():.4f}")
            print(f"  Min: {values.min():.4f}")
            print(f"  Max: {values.max():.4f}")

def export_results(df_final, config):
    """Exporte les résultats"""
    
    if len(df_final) == 0:
        print("Aucune donnée à exporter")
        return
    
    # Préparation des colonnes d'export
    export_columns = [
        'question_id', 'category', 'difficulty', 'question_similarity',
        'reference_question', 'actual_question', 'reference_answer', 'actual_answer',
        'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
        'trace_id', 'timestamp', 'session_id', 'user_id', 'trace_name',
        'latency', 'total_cost'
    ]
    
    # Filtrage des colonnes existantes
    available_columns = [col for col in export_columns if col in df_final.columns]
    
    # Export CSV
    df_export = df_final[available_columns].copy()
    df_export.to_csv(config['output_csv'], index=False, encoding='utf-8')
    
    print(f"Résultats exportés vers : {config['output_csv']}")
    print(f"Colonnes exportées : {len(available_columns)}")
    print(f"Lignes exportées : {len(df_export)}")
    
    # Export JSON détaillé
    json_output = config.get('output_json', config['output_csv'].replace('.csv', '_detailed.json'))
    
    json_data = {
        'evaluation_metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'total_reference_questions': len(df_final),
            'smartrag_project': config['smartrag_project_name'],
            'langfuse_url': config['langfuse_base_url']
        },
        'summary_metrics': {},
        'detailed_results': df_final.to_dict('records')
    }
    
    # Calcul des métriques de résumé
    ragas_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    
    for metric in ragas_metrics:
        if metric in df_final.columns and df_final[metric].notna().any():
            values = df_final[metric].dropna()
            json_data['summary_metrics'][metric] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'count': int(len(values))
            }
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Analyse détaillée exportée vers : {json_output}")

def main():
    """Fonction principale"""
    
    print("=== Évaluation SmartRAG + Langfuse avec Ragas ===")
    
    # 1. Chargement de la configuration
    config = load_config()
    print(f"Configuration chargée pour le projet: {config['smartrag_project_name']}")
    
    # 2. Connexion à Langfuse
    langfuse_client = connect_langfuse(config)
    print("Connexion à Langfuse réussie")
    
    # 3. Chargement des données de référence
    df_reference = load_reference_data(config['reference_csv'])
    
    # 4. Récupération des traces SmartRAG
    traces = fetch_smartrag_traces(langfuse_client, config)
    
    if not traces:
        print("Aucune trace trouvée. Vérifiez la configuration.")
        return
    
    # 5. Traitement des traces
    print("Traitement des traces SmartRAG...")
    processed_traces = []
    
    for trace in traces:
        try:
            processed = extract_rag_components(trace)
            if processed['question'] and processed['answer']:
                processed_traces.append(processed)
        except Exception as e:
            print(f"Erreur lors du traitement de la trace {trace.get('trace_id', 'unknown')}: {e}")
            continue
    
    if not processed_traces:
        print("Aucune trace valide trouvée après traitement")
        return
    
    df_traces = pd.DataFrame(processed_traces)
    print(f"Traces traitées avec succès : {len(df_traces)}")
    
    # 6. Correspondance des traces aux questions de référence
    df_matched = match_traces_to_reference(df_traces, df_reference)
    
    if len(df_matched) == 0:
        print("Aucune correspondance trouvée entre les traces et les questions de référence")
        return
    
    print(f"Correspondances trouvées : {len(df_matched)}/{len(df_reference)}")
    
    # 7. Évaluation avec Ragas
    df_final = evaluate_with_ragas(df_matched, config)
    
    # 8. Visualisations
    create_visualizations(df_final)
    
    # 9. Export des résultats
    export_results(df_final, config)
    
    print("\n=== Évaluation terminée ===")

if __name__ == "__main__":
    main()