#!/usr/bin/env python3
"""
Test direct de l'API v3 de Jira.
"""

import requests
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def test_api_v3_direct():
    """Test direct de l'API v3."""
    
    # Configuration
    jira_url = os.getenv("JIRA_URL")
    email = os.getenv("JIRA_EMAIL")
    api_token = os.getenv("JIRA_API_TOKEN")
    
    if not all([jira_url, email, api_token]):
        print("❌ Variables d'environnement manquantes")
        return
    
    # Headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Auth
    auth = (email, api_token)
    
    # URL
    search_url = f"{jira_url.rstrip('/')}/rest/api/3/search/jql"
    
    # Test 1: Recherche simple
    print("🔍 Test 1: Recherche simple")
    payload1 = {
        "jql": "created >= -7d",
        "maxResults": 3,
        "fields": ["key", "summary", "status", "created"]
    }
    
    print(f"URL: {search_url}")
    print(f"Payload: {payload1}")
    
    try:
        response = requests.post(search_url, json=payload1, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès: {data.get('total', 0)} tickets trouvés")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Recherche par projet
    print("🔍 Test 2: Recherche par projet")
    payload2 = {
        "jql": "project = SHARP ORDER BY created DESC",
        "maxResults": 3,
        "fields": ["key", "summary", "status", "created"]
    }
    
    print(f"URL: {search_url}")
    print(f"Payload: {payload2}")
    
    try:
        response = requests.post(search_url, json=payload2, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès: {data.get('total', 0)} tickets trouvés")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    # Test 3: Recherche plus large
    print("🔍 Test 3: Recherche plus large")
    payload3 = {
        "jql": "project = SHARP",  # Limité au projet SHARP
        "maxResults": 5,
        "fields": ["key", "summary", "status", "created"]
    }
    
    print(f"URL: {search_url}")
    print(f"Payload: {payload3}")
    
    try:
        response = requests.post(search_url, json=payload3, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès: {data.get('total', 0)} tickets trouvés")
            if data.get('issues'):
                for issue in data['issues'][:3]:
                    print(f"  - {issue.get('key')}: {issue.get('fields', {}).get('summary', 'No summary')}")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    # Test 4: Recherche des tickets récents (ceux que nous avons créés)
    print("🔍 Test 4: Recherche des tickets récents")
    payload4 = {
        "jql": "key in (SHARP-178, SHARP-179, SHARP-180)",  # Les tickets que nous avons créés
        "maxResults": 5,
        "fields": ["key", "summary", "status", "created"]
    }
    
    print(f"URL: {search_url}")
    print(f"Payload: {payload4}")
    
    try:
        response = requests.post(search_url, json=payload4, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès: {data.get('total', 0)} tickets trouvés")
            print(f"🔍 DEBUG: Structure de la réponse: {data}")
            if data.get('issues'):
                print(f"🔍 DEBUG: Nombre d'issues: {len(data['issues'])}")
                for i, issue in enumerate(data['issues']):
                    print(f"🔍 DEBUG: Issue {i}: {issue}")
                    print(f"  - {issue.get('key')}: {issue.get('fields', {}).get('summary', 'No summary')}")
            else:
                print("🔍 DEBUG: Aucune clé 'issues' dans la réponse")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    # Test 5: Lister les projets
    print("🔍 Test 5: Lister les projets")
    projects_url = f"{jira_url.rstrip('/')}/rest/api/3/project"
    
    try:
        response = requests.get(projects_url, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            projects = response.json()
            print(f"✅ Succès: {len(projects)} projets trouvés")
            for project in projects:
                print(f"  - {project.get('key')}: {project.get('name')}")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 6: Recherche avec les champs spécifiés
    print("🔍 Test 6: Recherche avec champs spécifiés")
    payload6 = {
        "jql": "key in (SHARP-178, SHARP-179, SHARP-180)",
        "maxResults": 5,
        "fields": ["key", "summary", "status", "created"]
    }
    
    print(f"URL: {search_url}")
    print(f"Payload: {payload6}")
    
    try:
        response = requests.post(search_url, json=payload6, headers=headers, auth=auth)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(data)
            print(f"✅ Succès: {data.get('total', 0)} tickets trouvés")
            if data.get('issues'):
                for issue in data['issues']:
                    fields = issue.get('fields', {})
                    print(f"  - {issue.get('key')}: {fields.get('summary', 'No summary')}")
        else:
            print(f"❌ Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_api_v3_direct()
