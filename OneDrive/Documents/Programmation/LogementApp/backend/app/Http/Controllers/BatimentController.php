<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class BatimentController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'id_client' => 'required|integer|exists:clients,id',
            'nom' => 'required|string|max:255',
            'type' => 'nullable|string|max:100',
            'adresse' => 'required|string|max:255',
            'ville' => 'required|string|max:100',
            'province' => 'required|string|max:100',
            'code_postal' => 'required|string|max:20',  
            'description' => 'nullable|string',
            'prix_moyen' => 'nullable|numeric',
            'latitude' => 'required|numeric',
            'longitude' => 'required|numeric',
            'images.*' => 'nullable|image|mimes:jpg,jpeg,png|max:2048',
            'date_debut_disponibilite' => 'nullable|date',
            'date_fin_disponibilite' => 'nullable|date',
        ]);

        $dataBatiment = $validated;
        unset($dataBatiment['images']);

        $batimentId = DB::table('batiments')->insertGetId($dataBatiment);

        if ($request->hasFile('images')) {
            foreach ($request->file('images') as $image) {
                $path = $image->store('batiments', 'public');
                DB::table('batiment_images')->insert([
                    'batiment_id' => $batimentId,
                    'image_path' => $path,
                    'created_at' => now(),
                    'updated_at' => now(),
                ]);
            }
        }

        return response()->json([
            'message' => 'Bâtiment créé avec succès',
            'id' => $batimentId
        ], 201);
    }



    // 📌 Voir les disponibilités et réservations d’un bâtiment
    public function getDisponibilites($id)
    {
        $batiment = DB::table('batiments')
            ->select('date_debut_disponibilite', 'date_fin_disponibilite')
            ->where('id', $id)
            ->first();

        if (!$batiment) {
            return response()->json(['error' => 'Bâtiment introuvable'], 404);
        }

        $reservations = DB::table('reservations')
            ->select('date_debut', 'date_fin')
            ->where('id_batiment', $id)
            ->get();

        $prix = DB::table('batiments')->select('prix_moyen')->where('id', $id)->first();

        return response()->json([
            'date_debut_disponibilite' => $batiment->date_debut_disponibilite,
            'date_fin_disponibilite'   => $batiment->date_fin_disponibilite,
            'reservations'             => $reservations,
            "prix"                     => $prix
        ]);
    }

    // 📌 Réserver un bâtiment
    public function reserver(Request $request, $id)
    {
        $request->validate([
            'id_client'  => 'required|exists:clients,id',
            'date_debut' => 'required|date',
            'date_fin'   => 'required|date|after:date_debut'
        ]);

        $batiment = DB::table('batiments')
            ->where('id', $id)
            ->first();

        if (!$batiment) {
            return response()->json(['error' => 'Bâtiment introuvable'], 404);
        }

        // Vérifie si la demande est dans la plage de disponibilité
        if (
            ($batiment->date_debut_disponibilite && $request->date_debut < $batiment->date_debut_disponibilite) ||
            ($batiment->date_fin_disponibilite && $request->date_fin > $batiment->date_fin_disponibilite)
        ) {
            return response()->json(['error' => 'Dates hors plage de disponibilité'], 400);
        }

        // Vérifie les conflits avec les réservations existantes
        $overlap = DB::table('reservations')
            ->where('id_batiment', $id)
            ->where(function ($query) use ($request) {
                $query->whereBetween('date_debut', [$request->date_debut, $request->date_fin])
                    ->orWhereBetween('date_fin', [$request->date_debut, $request->date_fin])
                    ->orWhere(function ($q) use ($request) {
                        $q->where('date_debut', '<=', $request->date_debut)
                          ->where('date_fin', '>=', $request->date_fin);
                    });
            })
            ->exists();

        if ($overlap) {
            return response()->json(['error' => 'Bâtiment déjà réservé sur cette période'], 400);
        }

        // Enregistre la réservation
        $reservationId = DB::table('reservations')->insertGetId([
            'id_client'   => $request->id_client,
            'id_batiment' => $id,
            'date_debut'  => $request->date_debut,
            'date_fin'    => $request->date_fin
        ]);

        return response()->json([
            'message' => 'Réservation effectuée avec succès',
            'id' => $reservationId
        ], 201);
    }

public function rechercher(Request $request)
{
    $typesAutorises = ['appartement 2 et demi','appartement 3 et demi', 'condo', 'maison', 'une chambre'];

    $query = DB::table('batiments');

    // Filtre par ville
    if ($request->filled('ville')) {
        $query->where('ville', $request->ville);
    }

    // Filtre par province
    if ($request->filled('province')) {
        $query->where('province', $request->province);
    }

    // Filtre par type
    if ($request->filled('type') && in_array(strtolower($request->type), $typesAutorises)) {
        $query->where('type', $request->type);
    }

    // Filtre par prix minimum
    if ($request->filled('prix_min')) {
        $query->where('prix_moyen', '>=', $request->prix_min);
    }

    // Filtre par prix maximum
    if ($request->filled('prix_max')) {
        $query->where('prix_moyen', '<=', $request->prix_max);
    }

    // Filtre par disponibilité
    if ($request->filled('date_debut') && $request->filled('date_fin')) {
        $query->where(function ($q) use ($request) {
            $q->whereDate('date_debut_disponibilite', '<=', $request->date_fin)
              ->whereDate('date_fin_disponibilite', '>=', $request->date_debut);
        })->orderBy('date_debut_disponibilite', 'asc');
    }

    // Exécution de la requête
    $batiments = $query->get();

    // Récupérer les IDs des bâtiments pour récupérer les images
    $batimentIds = $batiments->pluck('id')->toArray();

    // Récupérer toutes les images liées à ces bâtiments
    $images = DB::table('batiment_images')
        ->whereIn('batiment_id', $batimentIds)
        ->get()
        ->groupBy('batiment_id');

    // Ajouter les URLs complètes d’images à chaque bâtiment
    $batiments->transform(function ($batiment) use ($images) {
        $batimentImages = $images->get($batiment->id, collect());

        $batiment->images_urls = $batimentImages->map(function ($img) {
            return asset('storage/' . $img->image_path);
        })->toArray();

        return $batiment;
    });

    // Retourner le JSON avec la structure { success: true, data: [...] }
    return response()->json([
        'success' => true,
        'data' => $batiments
    ]);
}


public function getByClient($id_client)
{
    $batiments = DB::table('batiments')
        ->leftJoin('batiment_images', function($join) {
            $join->on('batiments.id', '=', 'batiment_images.batiment_id');
        })
        ->select(
            'batiments.id',
            'batiments.adresse',
            'batiments.ville',
            'batiments.province',
            'batiments.type',
            'batiments.prix_moyen',
            DB::raw('MIN(batiment_images.image_path) as image_path') // Première image
        )
        ->where('batiments.id_client', $id_client)
        ->groupBy(
            'batiments.id',
            'batiments.adresse',
            'batiments.ville',
            'batiments.province',
            'batiments.type',
            'batiments.prix_moyen'
        )
        ->get();

    return response()->json([
        'success' => true,
        'data' => $batiments
    ]);
}

public function show($id)
{
    $batiment = DB::table('batiments')->where('id', $id)->first();

    if (!$batiment) {
        return response()->json(['message' => 'Bâtiment non trouvé'], 404);
    }

    return response()->json([
        'success' => true,
        'batiment' => $batiment
    ]);
}

public function update(Request $request, $id)
{
    $validated = $request->validate([
        'nom' => 'required|string|max:255',
        'type' => 'nullable|string|max:100',
        'adresse' => 'required|string|max:255',
        'ville' => 'required|string|max:100',
        'province' => 'required|string|max:100',
        'code_postal' => 'required|string|max:20',  
        'description' => 'nullable|string',
        'prix_moyen' => 'nullable|numeric',
        'latitude' => 'required|numeric',
        'longitude' => 'required|numeric',
        'images.*' => 'nullable|image|mimes:jpg,jpeg,png|max:2048',
        'date_debut_disponibilite' => 'nullable|date',
        'date_fin_disponibilite' => 'nullable|date',
    ]);

    $dataBatiment = $validated;
    unset($dataBatiment['images']);

    DB::table('batiments')->where('id', $id)->update($dataBatiment);

    if ($request->hasFile('images')) {
        foreach ($request->file('images') as $image) {
            $path = $image->store('batiments', 'public');
            DB::table('batiment_images')->insert([
                'batiment_id' => $id,
                'image_path' => $path,
                'created_at' => now(),
                'updated_at' => now(),
            ]);
        }
    }

    return response()->json([
        'message' => 'Bâtiment modifié avec succès',
        'id' => $id
    ], 200);
}


public function getProprietaire($id)
{
    $proprietaire = DB::table('batiments')
        ->join('clients', 'batiments.id_client', '=', 'clients.id')
        ->where('batiments.id', $id)
        ->select('clients.id as id_client', 'clients.email')
        ->first();

    if (!$proprietaire) {
        return response()->json(['error' => 'Propriétaire non trouvé'], 404);
    }

    return response()->json([
        'id_client' => $proprietaire->id_client,
        'email' => $proprietaire->email
    ]);
}


}
