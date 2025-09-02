#!/usr/bin/env python3
"""
Script pour générer un hash de mot de passe Jupyter
"""

import getpass
from jupyter_server.auth import passwd

def generate_jupyter_password():
    print("🔐 Générateur de mot de passe Jupyter")
    print()
    
    # Demander le mot de passe
    password = getpass.getpass("Entrez votre mot de passe Jupyter : ")
    confirm = getpass.getpass("Confirmez le mot de passe : ")
    
    if password != confirm:
        print("❌ Les mots de passe ne correspondent pas!")
        return
    
    if len(password) < 4:
        print("❌ Le mot de passe doit faire au moins 4 caractères!")
        return
    
    # Générer le hash
    password_hash = passwd(password)
    
    print("\n✅ Hash généré avec succès!")
    print(f"🔑 Votre hash : {password_hash}")
    print()
    print("📝 Pour l'utiliser dans votre docker-compose :")
    print(f"c.ServerApp.password = '{password_hash}'")
    print()
    print("💡 Remplacez la ligne dans docker-compose-simple.yml")

if __name__ == "__main__":
    generate_jupyter_password()