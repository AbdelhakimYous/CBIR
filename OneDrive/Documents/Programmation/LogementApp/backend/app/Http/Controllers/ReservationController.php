<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class ReservationController extends Controller
{
    public function reserver(Request $request, $id)
    {
        // Validation
        $request->validate([
            'id_client'  => 'required|exists:clients,id',
            'date_debut' => 'required|date',
            'date_fin'   => 'required|date|after:date_debut'
        ]);

        // 1️⃣ Récupérer le bâtiment
        $batiment = DB::table('batiments')->where('id', $id)->first();
        if (!$batiment) {
            return response()->json(['error' => 'Bâtiment introuvable'], 404);
        }

        // 2️⃣ Vérifier la plage de disponibilité
        if (
            $request->date_debut < $batiment->date_debut_disponibilite ||
            $request->date_fin > $batiment->date_fin_disponibilite
        ) {
            return response()->json(['error' => 'Dates hors plage de disponibilité'], 400);
        }

        // 3️⃣ Vérifier chevauchement avec d’autres réservations
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

        // 4️⃣ Calculer prix_total
        $start = new \DateTime($request->date_debut);
        $end   = new \DateTime($request->date_fin);
        $diff  = $start->diff($end)->days;
        $prix_moyen = $diff * $batiment->prix_moyen;

        // 5️⃣ Insérer la réservation
        $idReservation = DB::table('reservations')->insertGetId([
            'id_client'   => $request->id_client,
            'id_batiment' => $id,
            'date_debut'  => $request->date_debut,
            'date_fin'    => $request->date_fin,
            'prix_moyen'  => $prix_moyen
        ]);

        return response()->json([
            'success' => true,
            'id_reservation' => $idReservation
        ], 201);
    }

    // Récupérer disponibilités avec prix
    public function disponibilites($id)
    {
        $batiment = DB::table('batiments')->where('id', $id)->first();
        if (!$batiment) {
            return response()->json(['error' => 'Bâtiment introuvable'], 404);
        }

        $reservations = DB::table('reservations')
            ->where('id_batiment', $id)
            ->get();

        return response()->json([
            'date_debut_disponibilite' => $batiment->date_debut_disponibilite,
            'date_fin_disponibilite'   => $batiment->date_fin_disponibilite,
            'prix_moyen'               => (float) $batiment->prix_moyen,
            'reservations'             => $reservations
        ]);
    }
}
