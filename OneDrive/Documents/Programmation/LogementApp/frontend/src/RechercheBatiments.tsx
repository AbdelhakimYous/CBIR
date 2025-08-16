import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

type Batiment = {
  id: number;
  nom: string;
  type: string | null;
  adresse: string;
  ville: string | null;
  province: string | null;
  code_postal: string;
  description: string | null;
  prix_moyen: number | null;
  latitude: number;
  longitude: number;
  images_urls?: string[];
  date_debut_disponibilite: string | null;
  date_fin_disponibilite: string | null;
};

export default function RechercheBatiments() {
  const navigate = useNavigate();

  const [filters, setFilters] = useState({
    ville: '',
    province: '',
    type: '',
    prix_min: '',
    prix_max: '',
    date_debut: '',
    date_fin: '',
  });

  const [resultats, setResultats] = useState<Batiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [erreur, setErreur] = useState<string | null>(null);

  const typesAutorises = ['appartement 2 et demi','appartement 3 et demi', 'condo', 'maison', 'une chambre'];
  const provinces = ['QC', 'ON', 'AB'];
  const villes = ['Montreal', 'Quebec', 'Toronto', 'Calgary', 'Ontario'];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErreur(null);

    const queryParams = new URLSearchParams();
    if (filters.ville) queryParams.append('ville', filters.ville);
    if (filters.province) queryParams.append('province', filters.province);
    if (filters.type && typesAutorises.includes(filters.type.toLowerCase())) queryParams.append('type', filters.type);
    if (filters.prix_min) queryParams.append('prix_min', filters.prix_min);
    if (filters.prix_max) queryParams.append('prix_max', filters.prix_max);
    if (filters.date_debut) queryParams.append('date_debut', filters.date_debut);
    if (filters.date_fin) queryParams.append('date_fin', filters.date_fin);

    try {
      const res = await fetch(`http://localhost:8000/api/batiments/rechercher?${queryParams.toString()}`);
      if (!res.ok) throw new Error('Erreur lors de la récupération des données');

      const data = await res.json();
      setResultats(Array.isArray(data.data) ? data.data : []);
    } catch (err: any) {
      setErreur(err.message || 'Erreur inconnue');
      setResultats([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Recherche avancée de bâtiments</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: '1rem', display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
        
        <select name="province" value={filters.province} onChange={handleChange}>
          <option value="">-- Province --</option>
          {provinces.map(p => <option key={p} value={p}>{p}</option>)}
        </select>

        <select name="ville" value={filters.ville} onChange={handleChange}>
          <option value="">-- Ville --</option>
          {villes.map(v => <option key={v} value={v}>{v}</option>)}
        </select>

        <select name="type" value={filters.type} onChange={handleChange}>
          <option value="">-- Type --</option>
          {typesAutorises.map(t => <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>)}
        </select>

        <input type="number" name="prix_min" placeholder="Prix min" value={filters.prix_min} onChange={handleChange} min="0" />
        <input type="number" name="prix_max" placeholder="Prix max" value={filters.prix_max} onChange={handleChange} min="0" />
        <input type="date" name="date_debut" value={filters.date_debut} onChange={handleChange} />
        <input type="date" name="date_fin" value={filters.date_fin} onChange={handleChange} />

        <button type="submit" disabled={loading}>{loading ? 'Recherche...' : 'Rechercher'}</button>
      </form>

      {erreur && <p style={{ color: 'red' }}>{erreur}</p>}

      <div>
        {resultats.length === 0 && !loading && <p>Aucun bâtiment trouvé</p>}

        {resultats.map(batiment => {
          const premiereImage = batiment.images_urls?.[0] || null;
          return (
            <div
              key={batiment.id}
              style={{
                border: '1px solid #ccc',
                marginBottom: '1rem',
                padding: '0.5rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
              onClick={() => navigate(`/batiments/${batiment.id}`)}
            >
              {premiereImage && (
                <img
                  src={premiereImage}
                  alt={batiment.nom || 'Bâtiment'}
                  style={{ width: 150, height: 100, objectFit: 'cover' }}
                />
              )}
              <div>
                <h3>{batiment.nom || 'Nom inconnu'} {batiment.type && `(${batiment.type})`}</h3>
                <p>{batiment.adresse || ''}, {batiment.ville || ''}, {batiment.province || ''}</p>
                <p>Prix moyen : {batiment.prix_moyen ? `${batiment.prix_moyen} $` : 'Non spécifié'}</p>
                {batiment.description && <p>{batiment.description}</p>}
                <p>
                  Disponibilité du {batiment.date_debut_disponibilite || 'N/A'} au {batiment.date_fin_disponibilite || 'N/A'}
                </p>  
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
