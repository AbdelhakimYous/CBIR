import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo
from scipy.spatial import distance
import streamlit as st

def extraction_signatures(chemin_dossier, descripteur):
    #print(chemin_dossier)
    liste_carac=[]
    for root,dirs,files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                path_relative=os.path.relpath(os.path.join(root,file))
                #print(f'Relative : {path_relative}')
                path=os.path.join(root,file)
                #print(f'Path : {path}')
                caracteristique=descripteur(path)
                print(f'{caracteristique}')
                class_name=os.path.dirname(path_relative)
                #print(f'{class_name}')
                liste_carac.append(caracteristique+[class_name,path_relative])
    Sigantures=np.array(liste_carac)
    np.save(f'Signatures_{descripteur.__name__}.npy', Sigantures)
    print('fini')
    #_{descripteur.__name__} je fais ca pour eviter que ca me leve une erreur de nom pas valide

def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1 - v2))
    return dist


def euclidenne(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist=np.sqrt(np.sum((v1 - v2) ** 2))
    return dist


def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist=np.max(np.abs(v1 - v2))
    return dist


def canberra(v1, v2):
    return distance.canberra(v1, v2)


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

def haralick_feat(chemin):
    data=cv2.imread(chemin,0)
    features=haralick(data).mean(0).tolist()
    features=[float (x) for x in features]
    return features
def bitdesk_feat(chemin):
    data=cv2.imread(chemin,0)
    features=bio_taxo(data)
    features=[float (x) for x in features]
    return features
 
def concat(chemin):
    return glcm(chemin)+haralick_feat(chemin)+bitdesk_feat(chemin)

def Recherche_img(bdd_signature, img_requete, distance, k=5):
    img_similaire = []
    
    for i, instance in enumerate(bdd_signature):
        carac, labelling, img_chemin = instance[:-2], instance[-2], instance[-1]
        
        if distance == 'euclidienne':
            dist = euclidenne(carac, img_requete)
        elif distance == 'manhattan':
            dist = manhattan(carac, img_requete)
        elif distance == 'chebyshev':
            dist = chebyshev(carac, img_requete)
        elif distance == 'canberra':
            dist = canberra(carac, img_requete)
        else:
            raise ValueError(f"Type de distance non reconnu : {distance}")
        
        img_similaire.append((dist, labelling, img_chemin))

    img_similaire = sorted(img_similaire, key=lambda x: x[0])

    return img_similaire[:k]


st.title("Recherche d'images par similarité")
chemin_img = 'animalsCbir/turtle/0a47b7d021.jpg'


#img = cv2.imread(chemin_img)

#if img is not None:
 #   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #  st.image(img_rgb, caption=f"Image : {chemin_img}", use_column_width=True)
choixDistance = st.selectbox(
    'Sélectionner le type de distance',
    ('euclidienne', 'manhattan', 'chebyshev')
)
choixDescripteur = st.selectbox(
    'Sélectionner le type de descripteur',
    ('GLCM', 'haralick', 'BiT', 'concat')
)
if st.button("Extraire les signatures"):
    if(choixDescripteur == 'GLCM'):
        extraction_signatures('animalsCbir', glcm)
    elif(choixDescripteur == 'haralick'):
        extraction_signatures('animalsCbir', haralick_feat)
    elif(choixDescripteur == 'BiT'):
        extraction_signatures('animalsCbir', bitdesk_feat)
    else:
        extraction_signatures('animalsCbir', concat)

imageUpload = st.file_uploader('Insérer une image', type=['png', 'jpg', 'jpeg'])

if imageUpload is not None:
    
    file_bytes = np.asarray(bytearray(imageUpload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    imageNG = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('imageUpload.png', imageNG)
    st.image(imageNG, channels='GRAY', caption='Image Upload')

    #caracteristique_reqete = concat(imageNG)
    #ca ne marche car concat veut  un chemin

    if(choixDescripteur == 'GLCM'):
        caracteristique_reqete = glcm('imageUpload.png')
        signatureGLCM = np.load('Signatures_glcm.npy', allow_pickle=True)

    elif(choixDescripteur == 'haralick'):
        caracteristique_reqete = haralick_feat('imageUpload.png')
        signatureGLCM = np.load('Signatures_haralick_feat.npy', allow_pickle=True)

    elif(choixDescripteur == 'BiT'):
        caracteristique_reqete = bitdesk_feat('imageUpload.png')
        signatureGLCM = np.load('Signatures_bitdesk_feat.npy', allow_pickle=True)
    else:
        caracteristique_reqete = concat("imageUpload.png")
        signatureGLCM = np.load('Signature.npy', allow_pickle=True)
    print(choixDescripteur)

    resultat = Recherche_img(
    bdd_signature=signatureGLCM, 
    img_requete=caracteristique_reqete, 
    distance=choixDistance, 
    k=5
)

    st.subheader("Résultats :")
    for i, x in enumerate(resultat):
        chemin_image = f'./{x[2]}' 

        img = cv2.imread(chemin_image)

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
        else:
            st.text(f"Image non trouvée : {chemin_image}")
 