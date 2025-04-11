import face_recognition
import numpy as np
import mysql.connector
import pickle
import cv2
import bcrypt
import streamlit as st
import os
import time 
from streamlit_extras.switch_page_button import switch_page


def glcm(chemin):
    data=cv2.imread(chemin,0)
    co_matrice=graycomatrix(data,[1],[0],None,symmetric=False,normed=False)
    contrast=float(graycoprops(co_matrice,'contrast')[0,0])
    dissimilarity=float(graycoprops(co_matrice,'dissimilarity')[0,0])
    correlation=float(graycoprops(co_matrice,'correlation')[0,0])
    homogeneity=float(graycoprops(co_matrice,'homogeneity')[0,0])
    ASM=float(graycoprops(co_matrice,'ASM')[0,0])
    energy=float(graycoprops(co_matrice,'energy')[0,0])
    return [contrast,dissimilarity,correlation,homogeneity,ASM,energy]

def euclidenne(v1,v2):
    v1=np.array(v1).astype('float')
    v2=np.array(v2).astype('float')
    dist=np.sqrt(np.sum(v1-v2)**2)
    return dist

def db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='reconnaissance_faciale'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Erreur de connexion bd : {err}")
        return None

def charger_descripteurs():
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nom_utilisateur, caracteristiques_facial FROM Utilisateur")
            utilisateurs = cursor.fetchall()
            descripteurs = []
            noms = []
            
            for utilisateur in utilisateurs:
                nom, descripteur_serialise = utilisateur
                if descripteur_serialise:  
                    descripteur = pickle.loads(descripteur_serialise) 
                    descripteurs.append(descripteur)
                    noms.append(nom)

            cursor.close()
            conn.close()
            return descripteurs, noms
    except mysql.connector.Error as err:
        print(f"Erreur des descripteurs : {err}")
        return [], []

def inserer_utilisateur(nom_utilisateur, email, mot_de_passe, type_auth, descripteurs_faciaux):
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            descripteurs_binaires = pickle.dumps(descripteurs_faciaux)  # Sérialiser les descripteurs
            query = """
                INSERT INTO Utilisateur (nom_utilisateur, email, mot_de_passe, type_authentification, caracteristiques_facial)
                VALUES (%s, %s, %s, %s, %s)
            """
            value = (nom_utilisateur, email, mot_de_passe, type_auth, descripteurs_binaires)
            cursor.execute(query, value)
            conn.commit()
            cursor.close()
            conn.close()
    except mysql.connector.Error as err:
        print(f" probleme ajout de l'utilisateur : {err}")

def verification_utilisateurSelonMotDePasse(nom_utilisateur, mot_de_passe):
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mot_de_passe FROM Utilisateur WHERE nom_utilisateur = %s", (nom_utilisateur,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            if user and bcrypt.checkpw(mot_de_passe.encode('utf-8'), user[0].encode('utf-8')):
                return True
            return False
    except mysql.connector.Error as err:
        print(f"Erreur lors de la vérification de l'utilisateur : {err}")
        return False

def verifier_descripteurs(nom_utilisateur):
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT caracteristiques_facial FROM Utilisateur WHERE nom_utilisateur = %s", (nom_utilisateur,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result and result[0]:
                descripteurs_faciaux = pickle.loads(result[0])
                return descripteurs_faciaux
            else:
                return None
    except mysql.connector.Error as err:
        print(f"Erreur lors de la récupération des descripteurs : {err}")
        return None

def hash_password(mot_de_passe):
    hashed = bcrypt.hashpw(mot_de_passe.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')

def comparer_descripteurs(descripteur_1, descripteur_2):
    if descripteur_1 is not None and descripteur_2 is not None and len(descripteur_1) == len(descripteur_2):
        dist = np.linalg.norm(np.array(descripteur_1) - np.array(descripteur_2))
        print(f"Distance entre descripteurs : {dist}")
        return dist
    else:
        return float('inf')

capture = cv2.VideoCapture(0)

descripteurs, noms = charger_descripteurs()

if descripteurs:
    # S'assurer que tous les descripteurs ont la même dimension sinon  ca va creer des problemes
    same_shape = all(len(desc) == len(descripteurs[0]) for desc in descripteurs)
    
    if same_shape:
        # Convertir descripteurs en tableau NumPy (2D)
        descripteurs = np.array(descripteurs)
        
        if descripteurs.ndim == 1:
            descripteurs = descripteurs.reshape(1, -1)  
    else:
        print("Les descripteurs ont des dimensions différentes.")
else:
    descripteurs = []

st.title("Reconnaissance Faciale et Gestion Utilisateur")

st.subheader("Inscription Utilisateur")
nom = st.text_input("Nom d'utilisateur")
email = st.text_input("Email")
mot_de_passe = st.text_input("Mot de passe", type="password")
type_auth = st.selectbox("Type d'authentification", ["Mot de passe", "Facial"])

if st.button("S'inscrire"):
    hashed_password = hash_password(mot_de_passe)
    
    if type_auth == "Facial":
        st.write("Veuillez vous positionner devant la caméra...")
        img_placeholder = st.empty()
        
        reponse,image=capture.read()
        if reponse:
            img_reduit=cv2.resize(image,(0,0),None,0.25,0.25)
        
            image_reduit=cv2.cvtColor(img_reduit,cv2.COLOR_BGR2RGB)            
            encodage=face_recognition.face_encodings(image_reduit)[0]
            encodage=encodage.tolist()
            #ici jai enleve +[nom] car ca ma cree des probleme avec npy liste de plus de 128 valeurs 
            #donc ca ma cree des probleme au niveau de la comparaison donc je l<ai enleve
            if encodage:
                st.success("Visage reconnu")                    
                inserer_utilisateur(nom, email, hashed_password, type_auth, encodage)
                st.success("Inscription avec succès ")
            else:
                st.error("Impossible extraire les caractéristiques")
        else:
                st.error("Aucun visage détecté.")
    else:
            st.error("Erreur d'accès à la caméra.")

st.subheader("Connexion Utilisateur")
nom_utilisateur_connexion = st.text_input("Nom d'utilisateur (Connexion)")
mot_de_passe_connexion = st.text_input("Mot de passe (Connexion)", type="password")

auth_mode = st.selectbox("Choisissez la méthode d'authentification", ["Mot de passe", "Reconnaissance faciale"])

if st.button("Se connecter"):
    if auth_mode == "Mot de passe":
        if verification_utilisateurSelonMotDePasse(nom_utilisateur_connexion, mot_de_passe_connexion):
            st.success("Connexion réussie")
        else:
            st.error("utilisateur ou mot de passe incorrect")
    
if auth_mode == "Reconnaissance faciale":
    st.text("Veuillez vous positionner devant la caméra.")
    capture = cv2.VideoCapture(0)  

    descripteurs_utilisateur = verifier_descripteurs(nom_utilisateur_connexion)
    
    if descripteurs_utilisateur is None:
        st.error(f"Aucun descripteur facial trouvé pour l'utilisateur {nom_utilisateur_connexion}")
    else:
        # ici je convertis  en tableau 2D
        if isinstance(descripteurs_utilisateur, list):
            descripteurs_utilisateur = np.array([descripteurs_utilisateur])
        elif isinstance(descripteurs_utilisateur, np.ndarray) and descripteurs_utilisateur.ndim == 1:
            descripteurs_utilisateur = descripteurs_utilisateur.reshape(1, -1)
        
        reponse, image = capture.read()
        capture.release() 
        cv2.destroyAllWindows()  

        if reponse:
            img_reduit = cv2.resize(image, (0, 0), None, 0.25, 0.25)
            image_reduit = cv2.cvtColor(img_reduit, cv2.COLOR_BGR2RGB)
            emplacement_face = face_recognition.face_locations(image_reduit)
            carac_face = face_recognition.face_encodings(image_reduit, emplacement_face)

            if carac_face:
                for encodage, loc in zip(carac_face, emplacement_face):
                    match=face_recognition.compare_faces(descripteurs_utilisateur,encodage)
                    distFace=face_recognition.face_distance(descripteurs_utilisateur,encodage)
                    minDist=np.argmin(distFace)
                    y1,x2,y2,x1=loc
                    y1,x2,y2,x1=4*y1,4*x2,4*y2,4*x1
                    dist = euclidenne(descripteurs_utilisateur[0], encodage)

                    if match[minDist]==True and dist < 0.6:
                        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                        nom=noms[minDist]
                        cv2.putText(image,nom_utilisateur_connexion,(x1,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        st.image(image, channels="BGR")
    
                        time.sleep(4)
                        st.markdown("""
                            <meta http-equiv="refresh" content="0; url=/app2" />
                            """, unsafe_allow_html=True) 
                    else:
                        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                        nom='Inconnue'
                        cv2.putText(image,nom,(x1,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) 
                        st.image(image, channels="BGR")
                        st.text("Visage non reconnu")

                           
                              
        else:
            st.error("Erreur.")
