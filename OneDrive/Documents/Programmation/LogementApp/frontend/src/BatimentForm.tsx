import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import './BatimentForm.css';

interface FormData {
  nom: string;
  type: string;
  adresse: string;
  ville: string;
  province: string;
  code_postal: string;
  description: string;
  prix_moyen: string;
  latitude: string;
  longitude: string;
  date_debut_disponibilite: string;
  date_fin_disponibilite: string;
  images: File[];
}

declare global {
  interface Window {
    initMap?: () => void;
    google?: typeof google;
  }
}

export default function BatimentForm() {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();

  const [formData, setFormData] = useState<FormData>({
    nom: "",
    type: "",
    adresse: "",
    ville: "",
    province: "",
    code_postal: "",
    description: "",
    prix_moyen: "",
    latitude: "",
    longitude: "",
    date_debut_disponibilite: "",
    date_fin_disponibilite: "",
    images: [],
  });

  const [isGoogleLoaded, setIsGoogleLoaded] = useState(false);
  const adresseRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);

  const apiKey = "AIzaSyAz0c-u2Y9kihbAsdpV6Qi3mu8BOQuuUGw"; // Clé Google

  // Charger Google Maps
  useEffect(() => {
    if (window.google?.maps?.places) {
      setIsGoogleLoaded(true);
      setupAutocomplete();
      return;
    }

    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places&callback=initMap`;
    script.async = true;
    script.defer = true;

    window.initMap = () => {
      setIsGoogleLoaded(true);
      setupAutocomplete();
    };

    document.head.appendChild(script);

    return () => { if (window.initMap) delete window.initMap; };
  }, []);

  const setupAutocomplete = () => {
    if (!adresseRef.current || !window.google?.maps?.places) return;

    if (autocompleteRef.current) {
      window.google.maps.event.clearInstanceListeners(autocompleteRef.current);
    }

    const autocomplete = new window.google.maps.places.Autocomplete(adresseRef.current, {
      types: ["address"],
      componentRestrictions: { country: "ca" },
      fields: ["address_components", "geometry", "formatted_address"]
    });

    autocompleteRef.current = autocomplete;

    autocomplete.addListener("place_changed", () => {
      const place = autocomplete.getPlace();
      if (!place.geometry?.location || !place.address_components) return;

      const lat = place.geometry.location.lat();
      const lng = place.geometry.location.lng();
      let ville = "", province = "", code_postal = "";

      place.address_components.forEach(comp => {
        const types = comp.types;
        if (types.includes("locality") || types.includes("sublocality")) ville = comp.long_name;
        if (types.includes("administrative_area_level_1")) province = comp.short_name;
        if (types.includes("postal_code")) code_postal = comp.long_name;
        if (types.includes("postal_code_suffix") && !code_postal) code_postal = comp.long_name;
      });

      setFormData(prev => ({
        ...prev,
        adresse: place.formatted_address || "",
        latitude: lat.toString(),
        longitude: lng.toString(),
        ville,
        province,
        code_postal,
      }));

      if (adresseRef.current) adresseRef.current.value = place.formatted_address || "";
    });
  };

  // Charger les données si modification
  useEffect(() => {
    if (!id) return;

    const fetchBatiment = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/batiments/${id}`);
        if (!res.ok) throw new Error("Impossible de récupérer le bâtiment");

        const text = await res.text();
        const data = text ? JSON.parse(text) : null;

        const b = data?.data || data?.batiment;
        if (!b) return;

        setFormData({
          nom: b.nom || "",
          type: b.type || "",
          adresse: b.adresse || "",
          ville: b.ville || "",
          province: b.province || "",
          code_postal: b.code_postal || "",
          description: b.description || "",
          prix_moyen: b.prix_moyen?.toString() || "",
          latitude: b.latitude?.toString() || "",
          longitude: b.longitude?.toString() || "",
          date_debut_disponibilite: b.date_debut_disponibilite || "",
          date_fin_disponibilite: b.date_fin_disponibilite || "",
          images: [],
        });

        if (adresseRef.current) adresseRef.current.value = b.adresse || "";
      } catch (err) {
        console.error(err);
        alert("Erreur lors du chargement du bâtiment");
      }
    };

    fetchBatiment();
  }, [id]);

  // Gestion changement inputs
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value, files } = e.target as HTMLInputElement;
    if (files && name === "images") {
      setFormData(prev => ({ ...prev, images: Array.from(files) }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  // Gestion du submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const data = new FormData();

    // Ajouter id_client depuis localStorage
    const idClient = localStorage.getItem("id_client");
    if (!idClient) {
      alert("ID client manquant dans le localStorage");
      return;
    }
    data.append("id_client", idClient);

    Object.entries(formData).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "") {
        if (key === "images" && Array.isArray(value)) {
          value.forEach(file => data.append("images[]", file));
        } else {
          data.append(key, value.toString());
        }
      }
    });

    try {
      const url = id
        ? `http://localhost:8000/api/batiments/${id}`
        : "http://localhost:8000/api/creerBatiment";
      const method = "POST";

      const res = await fetch(url, { method, body: data });
      const text = await res.text();
      let result = null;
      try { result = text ? JSON.parse(text) : null; } catch { console.warn("Pas de JSON à parser"); }

      if (!res.ok) {
        alert("Erreur : " + JSON.stringify(result?.errors || result || res.statusText));
        return;
      }

      alert(result?.message || (id ? "Bâtiment modifié !" : "Bâtiment créé !"));
      navigate("/batiments");
    } catch (err) {
      console.error(err);
      alert("Erreur réseau");
    }
  };

  return (
    <div className="form-wrapper">
      <div className="form-container">
        {!isGoogleLoaded && <div className="loading">Chargement Google Maps...</div>}
        <form onSubmit={handleSubmit} encType="multipart/form-data">
          <input type="text" name="nom" placeholder="Nom" value={formData.nom} onChange={handleChange} required />
          <select name="type" value={formData.type} onChange={handleChange} required>
            <option value="">Sélectionnez un type</option>
            <option value="appartement 2 et demi">Appartement 2 et demi</option>
            <option value="appartement 3 et demi">Appartement 3 et demi</option>
            <option value="condo">Condo</option>
            <option value="maison">Maison</option>
            <option value="une chambre">Une chambre</option>
          </select>
          <input type="text" name="adresse" placeholder="Adresse (Google Maps)" ref={adresseRef} onChange={handleChange} required />
          <input type="text" name="ville" placeholder="Ville" value={formData.ville} onChange={handleChange} required />
          <input type="text" name="province" placeholder="Province" value={formData.province} onChange={handleChange} required />
          <input type="text" name="code_postal" placeholder="Code postal" value={formData.code_postal} onChange={handleChange} required />
          <input type="number" step="0.0000001" name="latitude" placeholder="Latitude" value={formData.latitude} readOnly />
          <input type="number" step="0.0000001" name="longitude" placeholder="Longitude" value={formData.longitude} readOnly />
          <textarea name="description" placeholder="Description" value={formData.description} onChange={handleChange} />
          <input type="number" step="0.01" name="prix_moyen" placeholder="Prix moyen" value={formData.prix_moyen} onChange={handleChange} />
          <input type="date" name="date_debut_disponibilite" value={formData.date_debut_disponibilite} onChange={handleChange} />
          <input type="date" name="date_fin_disponibilite" value={formData.date_fin_disponibilite} onChange={handleChange} />
          <input type="file" name="images" onChange={handleChange} accept=".jpg,.jpeg,.png" multiple />
          <button type="submit" disabled={!isGoogleLoaded}>{id ? "Modifier" : "Ajouter"} bâtiment</button>
        </form>
      </div>
    </div>
  );
}
