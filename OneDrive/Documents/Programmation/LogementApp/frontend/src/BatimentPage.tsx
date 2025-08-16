import React, { useEffect, useState } from "react";
import axios from "axios";
import { PayPalScriptProvider, PayPalButtons } from "@paypal/react-paypal-js";
import { useParams, useNavigate } from "react-router-dom";

interface Reservation {
  id: number;
  date_debut: string;
  date_fin: string;
}

interface Disponibilite {
  date_debut_disponibilite: string;
  date_fin_disponibilite: string;
  prix: {
    prix_moyen: string;
  };
  reservations: Reservation[];
}

const BatimentPage: React.FC = () => {
  const { idBatiment } = useParams<{ idBatiment: string }>();
  const idClient = localStorage.getItem("id_client");
  const emailClient = localStorage.getItem("email");

  const [dispo, setDispo] = useState<Disponibilite | null>(null);
  const [loading, setLoading] = useState(true);
  const [dateDebut, setDateDebut] = useState("");
  const [dateFin, setDateFin] = useState("");
  const [message, setMessage] = useState("");
  const [paiementEffectue, setPaiementEffectue] = useState(false);
  const [reservationEnCours, setReservationEnCours] = useState(false);

  const [proprio, setProprio] = useState<{ id_client: number; email: string } | null>(null);

  const navigate = useNavigate();

  // Charger disponibilités
  useEffect(() => {
    if (!idBatiment) {
      setMessage("❌ ID bâtiment manquant");
      setLoading(false);
      return;
    }

    axios
      .get(`http://localhost:8000/api/batiments/${idBatiment}/disponibilites`)
      .then((res) => setDispo(res.data))
      .catch((err) => {
        console.error("Erreur chargement:", err);
        setMessage("❌ Erreur lors du chargement des disponibilités");
      })
      .finally(() => setLoading(false));
  }, [idBatiment]);

  // Charger propriétaire
  useEffect(() => {
    if (!idBatiment) return;

    axios
      .get(`http://localhost:8000/api/batiments/${idBatiment}/proprietaire`)
      .then((res) => {
        setProprio(res.data);
        console.log("Propriétaire :", res.data.email, res.data.id_client);
      })
      .catch((err) => console.error("Erreur récupération propriétaire :", err));
  }, [idBatiment]);

  // Calcul prix total
  const prixTotal = React.useMemo(() => {
    if (!dateDebut || !dateFin || !dispo) return 0;
    const start = new Date(dateDebut);
    const end = new Date(dateFin);
    const diffJours = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
    const prixParJour = parseFloat(dispo.prix.prix_moyen);
    return diffJours > 0 ? diffJours * prixParJour : 0;
  }, [dateDebut, dateFin, dispo]);

  // Vérifier disponibilité
  const verifierDisponibilite = () => {
    if (!dispo || !dateDebut || !dateFin) return false;

    if (dateDebut < dispo.date_debut_disponibilite || dateFin > dispo.date_fin_disponibilite) {
      return false;
    }

    const overlap = dispo.reservations.some((r) => {
      const rDebut = new Date(r.date_debut);
      const rFin = new Date(r.date_fin);
      const debut = new Date(dateDebut);
      const fin = new Date(dateFin);
      return (debut >= rDebut && debut <= rFin) || (fin >= rDebut && fin <= rFin) || (debut <= rDebut && fin >= rFin);
    });

    return !overlap;
  };

  const isDisponible = verifierDisponibilite();

  // Message de disponibilité
  const messageDisponibilite = React.useMemo(() => {
    if (paiementEffectue) return "";
    if (!dateDebut || !dateFin || !dispo) return "";

    if (dateDebut < dispo.date_debut_disponibilite || dateFin > dispo.date_fin_disponibilite) {
      return "❌ Dates hors plage de disponibilité";
    }

    const overlap = dispo.reservations.some((r) => {
      const rDebut = new Date(r.date_debut);
      const rFin = new Date(r.date_fin);
      const debut = new Date(dateDebut);
      const fin = new Date(dateFin);
      return (debut >= rDebut && debut <= rFin) || (fin >= rDebut && fin <= rFin) || (debut <= rDebut && fin >= rFin);
    });

    if (overlap) return "❌ Bâtiment déjà réservé sur cette période";
    return "✅ Dates disponibles";
  }, [dateDebut, dateFin, dispo, paiementEffectue]);

  // Fonction simplifiée pour envoyer juste le prix
  const envoyerEmailSimple = async () => {
    if (!emailClient) {
      console.error("Email du client non trouvé");
      setMessage("❌ Email client manquant");
      return;
    }

    try {
      console.log("Envoi email simple avec prix:", prixTotal);
      const response = await axios.post(
        "http://localhost:8000/api/envoyer-email-simple",
        { email_client: emailClient, prix: prixTotal },
        { headers: { "Content-Type": "application/json", Accept: "application/json" } }
      );
      console.log("✅ Email simple envoyé:", response.data);
      setMessage("✅ Email de confirmation envoyé !");
    } catch (error) {
      console.error("❌ Erreur email simple:", error);
      setMessage("❌ Erreur lors de l'envoi de l'email");
    }
  };

  // Fonction de réservation modifiée
  const reserver = async () => {
    if (!idBatiment || !idClient) {
      setMessage("❌ Données manquantes (ID bâtiment ou client)");
      return;
    }
    if (!dateDebut || !dateFin) {
      setMessage("❌ Veuillez sélectionner des dates");
      return;
    }

    setReservationEnCours(true);
    setMessage("🔄 Création de la réservation...");

    try {
      await axios.post(
        `http://localhost:8000/api/batiments/${idBatiment}/reserver`,
        { id_client: Number(idClient), date_debut: dateDebut, date_fin: dateFin },
        { headers: { "Content-Type": "application/json" } }
      );

      await envoyerEmailSimple();
      setPaiementEffectue(true);

      const res = await axios.get(`http://localhost:8000/api/batiments/${idBatiment}/disponibilites`);
      setDispo(res.data);
    } catch (err: any) {
      console.error("Erreur réservation:", err);
      const errorMsg = err.response?.data?.error || err.response?.data?.message || err.message || "Erreur lors de la réservation";
      setMessage(`❌ ${errorMsg}`);
    } finally {
      setReservationEnCours(false);
    }
  };

  const reserverAvecPayPal = async () => {
    await reserver();
  };

  // Redirection vers la page review
  const allerVersReview = () => {
    if (!proprio) return;
    navigate(`/review/${proprio.id_client}`);
  };

  if (loading) return <div style={{ padding: "20px" }}>🔄 Chargement...</div>;
  if (!dispo) return <div style={{ padding: "20px" }}>❌ Bâtiment introuvable</div>;

  return (
    <div style={{ padding: "20px", maxWidth: "800px" }}>
      <h1>🏢 Bâtiment #{idBatiment}</h1>

      {/* Disponibilités */}
      <div style={{ marginBottom: "20px" }}>
        <p>
          📅 Disponible du <strong>{dispo.date_debut_disponibilite}</strong> au{" "}
          <strong>{dispo.date_fin_disponibilite}</strong>
        </p>
        <p>💰 Prix par jour : <strong>{parseFloat(dispo.prix.prix_moyen)} CAD</strong></p>
      </div>

      {/* Sélection dates */}
      <div style={{ marginBottom: "20px" }}>
        <label style={{ marginRight: "15px" }}>
          <strong>Date début :</strong>
          <br />
          <input
            type="date"
            value={dateDebut}
            onChange={(e) => setDateDebut(e.target.value)}
            style={{ padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
          />
        </label>
        <label>
          <strong>Date fin :</strong>
          <br />
          <input
            type="date"
            value={dateFin}
            onChange={(e) => setDateFin(e.target.value)}
            style={{ padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
          />
        </label>
      </div>

      {/* Total */}
      {prixTotal > 0 && (
        <div style={{ backgroundColor: "#e8f5e8", padding: "15px", borderRadius: "5px", marginBottom: "15px" }}>
          <p style={{ margin: 0, fontSize: "18px" }}>
            💰 <strong>Total: {prixTotal} CAD</strong>
          </p>
        </div>
      )}

      {/* Messages */}
      {messageDisponibilite && (
        <div
          style={{
            padding: "10px",
            borderRadius: "5px",
            marginBottom: "15px",
            backgroundColor: messageDisponibilite.includes("✅") ? "#d4edda" : "#f8d7da",
            color: messageDisponibilite.includes("✅") ? "#155724" : "#721c24",
            border: `1px solid ${messageDisponibilite.includes("✅") ? "#c3e6cb" : "#f5c6cb"}`,
          }}
        >
          {messageDisponibilite}
        </div>
      )}

      {message && (
        <div
          style={{
            padding: "10px",
            borderRadius: "5px",
            marginBottom: "15px",
            backgroundColor: message.includes("✅") ? "#d4edda" : "#f8d7da",
            color: message.includes("✅") ? "#155724" : "#721c24",
            border: `1px solid ${message.includes("✅") ? "#c3e6cb" : "#f5c6cb"}`,
          }}
        >
          {message}
        </div>
      )}

      {/* Réservation */}
      <div style={{ marginBottom: "30px" }}>
        <button
          onClick={reserver}
          disabled={reservationEnCours || !dateDebut || !dateFin || !isDisponible}
          style={{
            backgroundColor: reservationEnCours ? "#6c757d" : "#28a745",
            color: "white",
            border: "none",
            borderRadius: "5px",
            padding: "12px 25px",
            fontSize: "16px",
            cursor: reservationEnCours ? "not-allowed" : "pointer",
            marginRight: "10px",
          }}
        >
          {reservationEnCours ? "⏳ Réservation..." : "✅ Réserver (Test avec email simple)"}
        </button>

        {prixTotal > 0 && !paiementEffectue && isDisponible && (
          <div style={{ marginTop: "15px" }}>
            <PayPalScriptProvider
              options={{
                clientId: "ATjME8N8nnUNPhtCWvElbZ8IlsD8WidId2Poa_TZxEYttm9SHoalXC2uDlKaFcYLShA7hLIgbAB1dC-q",
                currency: "CAD",
                locale: "fr_CA",
              }}
            >
              <PayPalButtons
                style={{ layout: "vertical", color: "blue", height: 40 }}
                createOrder={(data, actions) =>
                  actions.order.create({
                    intent: "CAPTURE",
                    purchase_units: [
                      { amount: { currency_code: "CAD", value: prixTotal.toFixed(2) }, description: `Réservation bâtiment ${idBatiment}` },
                    ],
                  })
                }
                onApprove={(data, actions) =>
                  actions.order.capture().then(async () => {
                    await reserverAvecPayPal();
                  })
                }
                onError={(err) => setMessage("❌ Erreur avec le paiement PayPal")}
                onCancel={() => setMessage("⚠️ Paiement annulé")}
              />
            </PayPalScriptProvider>
          </div>
        )}

        {paiementEffectue && (
          <>
            <p style={{ color: "green", fontWeight: "bold" }}>✅ Paiement effectué et emails envoyés</p>

            {/* BOUTON REVIEW */}
            {proprio && (
              <div style={{ marginTop: "20px" }}>
                <button
                  onClick={allerVersReview}
                  style={{
                    backgroundColor: "#ffc107",
                    color: "#212529",
                    border: "none",
                    borderRadius: "5px",
                    padding: "10px 20px",
                    cursor: "pointer",
                  }}
                >
                  ✍️ Laisser un avis
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Réservations existantes */}
      <div>
        <h3>📋 Réservations existantes :</h3>
        {dispo.reservations.length > 0 ? (
          <ul>
            {dispo.reservations.map((r) => (
              <li key={r.id} style={{ color: "red", marginBottom: "5px" }}>
                ❌ <strong>Du {r.date_debut} au {r.date_fin}</strong>
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ color: "green" }}>✅ Aucune réservation</p>
        )}
      </div>
    </div>
  );
};

export default BatimentPage;
