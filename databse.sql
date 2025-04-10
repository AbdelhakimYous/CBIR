CREATE DATABASE reconnaissance_faciale;

USE reconnaissance_faciale;

-- Table pour stocker les informations des utilisateurs
CREATE TABLE Utilisateur (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nom_utilisateur VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    mot_de_passe VARCHAR(255) NOT NULL,
    type_authentification VARCHAR(50) NOT NULL,  -- Remplacement de ENUM par VARCHAR
    caracteristiques_facial LONGBLOB NOT NULL,
    UNIQUE (email),
    UNIQUE (nom_utilisateur)
);

-- Table pour stocker les logs des connexions (optionnel, si besoin)
CREATE TABLE Log_Connexion (
    id INT AUTO_INCREMENT PRIMARY KEY,
    utilisateur_id INT NOT NULL,
    date_connexion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    succes BOOLEAN,
    FOREIGN KEY (utilisateur_id) REFERENCES Utilisateur(id)
);
