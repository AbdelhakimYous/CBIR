import face_recognition
import numpy as np
import mysql.connector
import pickle
import cv2
import bcrypt
import streamlit as st

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
        print(f"Erreur lors de la connexion à la base de données : {err}")
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
        print(f"Erreur lors du chargement des descripteurs : {err}")
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
        print(f"Erreur lors de l'ajout de l'utilisateur : {err}")

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
nom_utilisateur = st.text_input("Nom d'utilisateur")
email = st.text_input("Email")
mot_de_passe = st.text_input("Mot de passe", type="password")
type_auth = st.selectbox("Type d'authentification", ["Mot de passe", "Facial"])

if st.button("S'inscrire"):
    hashed_password = hash_password(mot_de_passe)
    
    if type_auth == "Facial":
        st.write("Veuillez vous positionner devant la caméra...")
        img_placeholder = st.empty()
        
        ret, frame = capture.read()
        if ret:
            img_placeholder.image(frame, channels="BGR")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    descripteurs_faciaux = face_encodings[0]
                    st.success("Visage capturé avec succès!")
                    
                    inserer_utilisateur(nom_utilisateur, email, hashed_password, type_auth, descripteurs_faciaux)
                    st.success("Inscription avec succès ")
                else:
                    st.error("Impossible d'extraire les caractéristiques faciales. Veuillez réessayer.")
            else:
                st.error("Aucun visage détecté. Veuillez vous rapprocher de la caméra et réessayer.")
        else:
            st.error("Erreur d'accès à la caméra.")
    else:
        descripteurs_faciaux = [] 
        inserer_utilisateur(nom_utilisateur, email, hashed_password, type_auth, descripteurs_faciaux)
        st.success("Utilisateur inscrit avec succès")

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
                    distFace = face_recognition.face_distance(descripteurs_utilisateur, encodage)

                    if distFace[0] < 0.6: 
                        y1, x2, y2, x1 = loc
                        y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, nom_utilisateur_connexion, (x1, y2 + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        st.success(f"Reconnaissance réussie pour {nom_utilisateur_connexion}")
                        #st.session_state.page = 'app2'  # Mettre à jour la session
                        #st.switch_page("app2")
                        #ne maarche car bug
                    else:
                        y1, x2, y2, x1 = loc
                        y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(image, 'Inconnu', (x1, y2 + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        st.warning("Visage inconnu")
            
            st.image(image, channels="BGR")
        else:
            st.error("Erreur.")

