<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Mail;
use Illuminate\Support\Facades\DB;

class MailController extends Controller
{
    public function envoyerConfirmation(Request $request)
    {
        $data = $request->validate([
            'email_client' => 'required|email',
            'id_batiment' => 'required|integer',
            'montant' => 'required|numeric',
            'date_debut' => 'required|date',
            'date_fin' => 'required|date',
        ]);

        // Récupérer l'email du propriétaire
        $emailProprietaire = $this->getProprietaireEmail($data['id_batiment']);
        
        if (!$emailProprietaire) {
            return response()->json(['error' => 'Propriétaire non trouvé'], 404);
        }

        try {
            // Email au client
            Mail::raw(
                "Bonjour,\n\nVotre réservation a été confirmée avec succès !\n\n" .
                "Détails de votre réservation :\n" .
                "- Bâtiment : #{$data['id_batiment']}\n" .
                "- Date de début : {$data['date_debut']}\n" .
                "- Date de fin : {$data['date_fin']}\n" .
                "- Montant payé : {$data['montant']} CAD\n\n" .
                "Merci pour votre confiance !\n\nCordialement,\nL'équipe de réservation",
                function ($message) use ($data) {
                    $message->to($data['email_client'])
                            ->subject("✅ Confirmation de votre réservation - Bâtiment #{$data['id_batiment']}");
                }
            );

            // Email au propriétaire
            Mail::raw(
                "Bonjour,\n\nVous avez reçu une nouvelle réservation !\n\n" .
                "Détails de la réservation :\n" .
                "- Bâtiment : #{$data['id_batiment']}\n" .
                "- Client : {$data['email_client']}\n" .
                "- Date de début : {$data['date_debut']}\n" .
                "- Date de fin : {$data['date_fin']}\n" .
                "- Montant : {$data['montant']} CAD\n\n" .
                "Cordialement,\nL'équipe de réservation",
                function ($message) use ($emailProprietaire, $data) {
                    $message->to($emailProprietaire)
                            ->subject("🏢 Nouvelle réservation - Bâtiment #{$data['id_batiment']}");
                }
            );

            return response()->json([
                'message' => 'Emails envoyés avec succès',
                'email_client' => $data['email_client'],
                'email_proprietaire' => $emailProprietaire
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Erreur lors de l\'envoi des emails',
                'details' => $e->getMessage()
            ], 500);
        }
    }

    private function getProprietaireEmail($idBatiment)
    {
        return DB::table('batiments')
            ->join('clients', 'batiments.client_id', '=', 'clients.id')
            ->where('batiments.id', $idBatiment)
            ->value('clients.email');
    }

    public function getProprietaire($id)
    {
        $email = $this->getProprietaireEmail($id);

        if (!$email) {
            return response()->json(['error' => 'Propriétaire non trouvé'], 404);
        }

        return response()->json(['email' => $email]);
    }
}