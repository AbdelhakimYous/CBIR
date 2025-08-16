<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\BatimentController;
use App\Http\Controllers\AuthController;
use App\Http\Controllers\ReservationController;
use App\Http\Controllers\MailController;

Route::post('/creerBatiment', [BatimentController::class, 'store']);
Route::get('/batiments/{id}/disponibilites', [BatimentController::class, 'getDisponibilites']);


Route::get('/batiments/{id}/disponibilite', [ReservationController::class, 'disponibilite']);
Route::post('/batiments/{id}/reserver', [ReservationController::class, 'reserver']);

Route::get('/batiments/rechercher', [BatimentController::class, 'rechercher']);


Route::post('/signup', [AuthController::class, 'signup']);
Route::post('/login', [AuthController::class, 'login']);

Route::get('/batiments/client/{id_client}', [BatimentController::class, 'getByClient']);

Route::get('/batiments/{id}', [BatimentController::class, 'show']); // récupérer un bâtiment
Route::post('/batiments/{id}', [BatimentController::class, 'update']); // POST avec _method=PUT


Route::get('/batiments/{id}/proprietaire', [BatimentController::class, 'getProprietaire']);
use Illuminate\Support\Facades\Mail;


Route::get('/test-email', function () {
    try {
        Mail::raw('Test email depuis Laravel - Configuration OK', function ($message) {
            $message->to('mryousabdelhakim@gmail.com')
                    ->subject('Test Email Configuration');
        });
        
        return response()->json(['message' => 'Email de test envoyé avec succès']);
    } catch (\Exception $e) {
        return response()->json([
            'error' => 'Erreur lors de l\'envoi du test',
            'details' => $e->getMessage()
        ], 500);
    }
});

// 2. Route de debug pour vérifier la réception des données
Route::post('/debug-confirmation', function (Request $request) {
    return response()->json([
        'message' => 'Données reçues correctement',
        'data' => $request->all(),
        'headers' => $request->headers->all()
    ]);
});

// 3. Route simple pour envoyer email avec juste le prix
Route::post('/envoyer-email-simple', function (Request $request) {
    try {
        $prix = $request->input('prix', 0);
        $emailClient = $request->input('email_client');
        
        if (!$emailClient) {
            return response()->json(['error' => 'Email client manquant'], 400);
        }

        if (!filter_var($emailClient, FILTER_VALIDATE_EMAIL)) {
            return response()->json(['error' => 'Email client invalide'], 400);
        }

        // Message simple avec juste le prix
        $message = "Votre réservation est confirmée. Prix total: {$prix} CAD";
        
        Mail::raw($message, function ($mail) use ($emailClient) {
            $mail->to($emailClient)
                 ->subject('Confirmation de réservation');
        });

        return response()->json([
            'success' => true,
            'message' => 'Email envoyé avec succès',
            'prix_envoye' => $prix,
            'email_envoye_a' => $emailClient
        ]);
        
    } catch (\Exception $e) {
        return response()->json([
            'success' => false,
            'error' => 'Erreur lors de l\'envoi de l\'email',
            'details' => $e->getMessage()
        ], 500);
    }
});

use App\Http\Controllers\ReviewController;

Route::get('/reviews/{id_client}', [ReviewController::class, 'getByExpediteur']);
Route::post('/reviews/envoyer', [ReviewController::class, 'envoyerCommentaire']);
