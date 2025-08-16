import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";

interface Review {
  id: number;
  id_client_expediteur: number;
  id_client_destinataire: number;
  contenu: string;
  created_at: string;
}

interface Proprio {
  id_client: number;
  email: string;
}

const ReviewPage: React.FC = () => {
  const { idBatiment } = useParams<{ idBatiment: string }>();
  const idClient = localStorage.getItem("id_client"); // ID du client connecté
  const [proprio, setProprio] = useState<Proprio | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [nouveauMessage, setNouveauMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [envoiEnCours, setEnvoiEnCours] = useState(false);

  // Récupérer le propriétaire du bâtiment
  useEffect(() => {
    if (!idBatiment) return;

    axios
      .get(`http://localhost:8000/api/batiments/${idBatiment}/proprietaire`)
      .then((res) => setProprio(res.data))
      .catch((err) => console.error("Erreur récupération propriétaire :", err));
  }, [idBatiment]);

  // Récupérer les messages existants
  useEffect(() => {
    if (!idClient) return;

    axios
      .get(`http://localhost:8000/api/reviews/${idClient}`)
      .then((res) => {
        setReviews(res.data);
      })
      .catch((err) => console.error("Erreur chargement reviews:", err))
      .finally(() => setLoading(false));
  }, [idClient]);

  // Envoyer un nouveau message
  const envoyerMessage = async () => {
    if (!nouveauMessage.trim() || !idClient || !proprio) return;
    setEnvoiEnCours(true);

    try {
      const res = await axios.post("http://localhost:8000/api/reviews/envoyer", {
        id_client_expediteur: Number(idClient),
        id_client_destinataire: Number(proprio.id_client),
        contenu: nouveauMessage,
      });

      // Ajouter le nouveau message en tête de liste
      setReviews((prev) => [
        { id: res.data.id, id_client_expediteur: Number(idClient), id_client_destinataire: proprio.id_client, contenu: nouveauMessage, created_at: new Date().toISOString() },
        ...prev,
      ]);
      setNouveauMessage("");
    } catch (err) {
      console.error("Erreur envoi message:", err);
    } finally {
      setEnvoiEnCours(false);
    }
  };

  if (loading) return <div>🔄 Chargement des messages...</div>;

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h1>💬 Messages avec {proprio?.email || "propriétaire"}</h1>

      <div style={{ marginBottom: "20px" }}>
        {reviews.length === 0 ? (
          <p>📭 Aucun message pour l’instant</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0 }}>
            {reviews.map((r) => (
              <li
                key={r.id}
                style={{
                  marginBottom: "10px",
                  padding: "10px",
                  borderRadius: "5px",
                  backgroundColor:
                    r.id_client_expediteur === Number(idClient)
                      ? "#d4edda"
                      : "#f8d7da",
                  color:
                    r.id_client_expediteur === Number(idClient)
                      ? "#155724"
                      : "#721c24",
                }}
              >
                <strong>
                  {r.id_client_expediteur === Number(idClient)
                    ? "Vous"
                    : proprio?.email}
                </strong>{" "}
                <span style={{ float: "right", fontSize: "12px" }}>
                  {new Date(r.created_at).toLocaleString()}
                </span>
                <p style={{ margin: "5px 0 0" }}>{r.contenu}</p>
              </li>
            ))}
          </ul>
        )}
      </div>

      {proprio && (
        <div style={{ marginTop: "20px" }}>
          <textarea
            value={nouveauMessage}
            onChange={(e) => setNouveauMessage(e.target.value)}
            placeholder="Écrire un message..."
            rows={3}
            style={{ width: "100%", padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }}
          />
          <button
            onClick={envoyerMessage}
            disabled={envoiEnCours || !nouveauMessage.trim()}
            style={{
              marginTop: "10px",
              padding: "10px 20px",
              backgroundColor: envoiEnCours ? "#6c757d" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: envoiEnCours ? "not-allowed" : "pointer",
            }}
          >
            {envoiEnCours ? "⏳ Envoi..." : "📩 Envoyer"}
          </button>
        </div>
      )}
    </div>
  );
};

export default ReviewPage;
