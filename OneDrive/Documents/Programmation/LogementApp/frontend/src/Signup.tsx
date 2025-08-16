import React, { useState } from 'react';
import './Signup.css';

export default function Signup() {
  const [nom, setNom] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('voyageur');

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const res = await fetch('http://127.0.0.1:8000/api/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          nom,
          email,
          password,
          role  // <--- rôle inclus
        })
      });

      const data = await res.json();

      if (data.success) {
        localStorage.setItem('id_client', data.id_client);
        localStorage.setItem('role', role);
        alert('Inscription réussie !');
      } else {
        alert('Erreur : ' + (data.message || 'Impossible de créer le compte'));
      }
    } catch (error) {
      console.error(error);
      alert('Erreur lors de la requête');
    }
  };

  return (
    <div className="signup-wrapper">
      <div className="form-container">
        <h2>Créer un compte</h2>
        <form onSubmit={handleSignup}>
          <div>
            <label htmlFor="nom">Nom</label>
            <input
              type="text"
              id="nom"
              placeholder="Votre nom"
              value={nom}
              onChange={(e) => setNom(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              placeholder="Votre email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="password">Mot de passe</label>
            <input
              type="password"
              id="password"
              placeholder="Votre mot de passe"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="role">Rôle</label>
            <select
              id="role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              required
            >
              <option value="voyageur">Voyageur</option>
              <option value="hote">Hôte</option>
            </select>
          </div>

          <button type="submit">S’inscrire</button>
        </form>

        <p className="login-link">
          Déjà un compte ? <a href="/login">Se connecter</a>
        </p>
      </div>
    </div>
  );
}
