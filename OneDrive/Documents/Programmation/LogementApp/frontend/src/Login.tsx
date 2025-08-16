import React, { useState } from 'react';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const res = await fetch('http://127.0.0.1:8000/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ email, password })
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert(errorData.message || 'Email ou mot de passe invalide');
        return;
      }

      const data = await res.json();
      localStorage.setItem('id_client', data.id_client);
      localStorage.setItem('role', data.role);
      localStorage.setItem('email', data.email);
      console.log(data.role);
      console.log(data.email);
      alert(data.message || 'Connexion réussie !');
    } catch (error) {
      console.error(error);
      alert('Erreur lors de la connexion');
    }
  };

  return (
    <form onSubmit={handleLogin}>
      <h2>Connexion</h2>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        required
      />
      <input
        type="password"
        placeholder="Mot de passe"
        value={password}
        onChange={e => setPassword(e.target.value)}
        required
      />
      <button type="submit">Se connecter</button>
    </form>
  );
}
