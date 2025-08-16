<?php

    namespace App\Http\Controllers;

    use Illuminate\Http\Request;
    use Illuminate\Support\Facades\DB;

    class ReviewController extends Controller
    {
        // Récupérer tous les commentaires envoyés par un client
        public function getByExpediteur($id_client)
        {
            $reviews = DB::table('reviews')
                ->where('id_client_expediteur', $id_client)
                ->orderBy('created_at', 'desc')
                ->get();

            return response()->json($reviews);
        }

        // Envoyer un commentaire
        public function envoyerCommentaire(Request $request)
        {
            $validated = $request->validate([
                'id_client_expediteur' => 'required|integer|exists:clients,id',
                'id_client_destinataire' => 'required|integer|exists:clients,id',
                'contenu' => 'required|string',
            ]);

            $id = DB::table('reviews')->insertGetId([
                'id_client_expediteur' => $validated['id_client_expediteur'],
                'id_client_destinataire' => $validated['id_client_destinataire'],
                'contenu' => $validated['contenu'],
                'created_at' => now(),
                'updated_at' => now(),
            ]);

            return response()->json([
                'message' => 'Commentaire envoyé',
                'id' => $id
            ], 201);
        }
    }
