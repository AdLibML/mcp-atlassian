#!/usr/bin/env python3
"""
Script de test pour créer un ticket Jira avec l'API v3.
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp_atlassian.jira import JiraFetcher, JiraConfig

# Charger les variables d'environnement
load_dotenv()

async def test_create_ticket():
    """Test de création d'un ticket Jira."""
    
    # Configuration Jira
    config = JiraConfig.from_env()
    jira = JiraFetcher(config=config)
    
    try:
        # Test 1: Lister les projets disponibles
        print("🔍 Récupération des projets...")
        projects = jira.get_all_projects()
        print(f"✅ Projets trouvés: {len(projects)}")
        
        if not projects:
            print("❌ Aucun projet trouvé. Vérifiez vos permissions.")
            return
            
        # Utiliser le premier projet disponible
        # projects[0] est un dictionnaire, pas un objet
        project_key = projects[0]["key"] if isinstance(projects[0], dict) else projects[0].key
        print(f"📋 Utilisation du projet: {project_key}")
        
        # Test 2: Créer un ticket de test
        print("\n🎫 Création d'un ticket de test...")
        
        ticket_data = {
            "project_key": project_key,
            "summary": "Test API v3 - Ticket créé par script",
            "description": "Ce ticket a été créé pour tester l'API v3 de Jira.\n\n**Fonctionnalités testées:**\n- Création de ticket\n- Format ADF pour la description\n- API v3",
            "issue_type": "Task",
            "assignee": None,  # Pas d'assignation
        }
        
        # Créer le ticket
        result = jira.create_issue(**ticket_data)
        
        if result and hasattr(result, 'key'):
            print(f"✅ Ticket créé avec succès: {result.key}")
            print(f"🔗 URL: {config.url}/browse/{result.key}")
        else:
            print("❌ Échec de la création du ticket")
            print(f"Résultat: {result}")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

async def test_search_tickets():
    """Test de recherche de tickets."""
    
    config = JiraConfig.from_env()
    jira = JiraFetcher(config=config)
    
    try:
        print("\n🔍 Test de recherche de tickets...")
        
        # D'abord récupérer les projets pour limiter la recherche
        projects = jira.get_all_projects()
        if not projects:
            print("❌ Aucun projet trouvé pour la recherche")
            return
            
        project_key = projects[0]["key"] if isinstance(projects[0], dict) else projects[0].key
        print(f"📋 Recherche dans le projet: {project_key}")
        
        # Rechercher les tickets récents dans un projet spécifique
        search_result = jira.search_issues(
            jql=f"project = {project_key} ORDER BY created DESC",
            limit=5
        )
        
        print(f"✅ {search_result.total} tickets trouvés")
        
        for issue in search_result.issues[:3]:  # Afficher les 3 premiers
            print(f"  - {issue.key}: {issue.summary}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la recherche: {e}")
        import traceback
        traceback.print_exc()

async def test_simple_search():
    """Test de recherche simple avec JQL basique."""
    
    config = JiraConfig.from_env()
    jira = JiraFetcher(config=config)
    
    try:
        print("\n🔍 Test de recherche simple...")
        
        # Recherche très simple
        search_result = jira.search_issues(
            jql="created >= -7d",  # Tickets créés dans les 7 derniers jours
            limit=3
        )
        
        print(f"✅ {search_result.total} tickets trouvés (7 derniers jours)")
        
        for issue in search_result.issues:
            print(f"  - {issue.key}: {issue.summary}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la recherche simple: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Fonction principale."""
    print("🚀 Test de l'API Jira v3")
    print("=" * 50)
    
    # Vérifier les variables d'environnement
    required_vars = ["JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Variables d'environnement manquantes: {missing_vars}")
        print("Veuillez configurer votre fichier .env")
        return
    
    print("✅ Variables d'environnement configurées")
    
    # Tests
    await test_simple_search()
    await test_search_tickets()
    await test_create_ticket()
    
    print("\n🎉 Tests terminés!")

if __name__ == "__main__":
    asyncio.run(main())
