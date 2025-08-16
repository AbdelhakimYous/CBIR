import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface Batiment {
  id: number;
  adresse: string;
  ville: string | null;
  province: string | null;
  type: string | null;
  prix_moyen: number | string | null;
  image_path: string | null;
}

const BatimentsList = () => {
  const [batiments, setBatiments] = useState<Batiment[]>([]);
  const idClient = localStorage.getItem('id_client');
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/api/batiments/client/${idClient}`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setBatiments(data.data);
          console.log("📦 Données reçues :", data.data);
        }
      })
      .catch(err => console.error("❌ Erreur API :", err));
  }, [idClient]);

  const onEditBatiment = (id: number) => {
    navigate(`/BatimentForm/${id}`);
  };

  return (
    <div style={{ display: 'grid', gap: '20px', padding: '20px' }}>
      {batiments.map(b => (
        <div
          key={b.id}
          style={{
            border: '1px solid #ccc',
            borderRadius: '8px',
            overflow: 'hidden',
            background: '#fff',
            boxShadow: '0 2px 6px rgba(0,0,0,0.1)'
          }}
        >
          {b.image_path && (
            <img
              src={`http://127.0.0.1:8000/storage/${b.image_path}`}
              alt={b.adresse || "Image bâtiment"}
              style={{ width: '100%', height: '200px', objectFit: 'cover' }}
            />
          )}
          <div style={{ padding: '10px', color: '#333' }}>
            <h3 style={{ margin: '0 0 10px 0' }}>{b.type || 'Type non précisé'}</h3>
            <p style={{ margin: '0 0 5px 0' }}>{b.adresse || 'Adresse non spécifiée'}</p>
            <p style={{ margin: '0 0 5px 0' }}>{(b.ville || 'Ville inconnue')}, {(b.province || 'Province inconnue')}</p>
            <p style={{ fontWeight: 'bold', color: '#007BFF' }}>
              {b.prix_moyen ? `${b.prix_moyen} $` : 'Prix non spécifié'}
            </p>
            {/* Bouton Modifier */}
            <button
              onClick={() => onEditBatiment(b.id)}
              style={{
                marginTop: '10px',
                padding: '8px 12px',
                backgroundColor: '#28a745',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Modifier
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default BatimentsList;

