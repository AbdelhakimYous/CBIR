<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class AuthController extends Controller
{
    public function signup(Request $request)
    {
        $request->validate([
            'nom' => 'required|string',
            'email' => 'required|email|unique:clients,email',
            'role' => 'required|string',
            'password' => 'required|string|min:6',
        ]);

        $id = DB::table('clients')->insertGetId([
            'nom' => $request->nom,
            'email' => $request->email,
            'role' => $request->role,
            'password' => Hash::make($request->password),
            'created_at' => now(),
            'updated_at' => now(),
        ]);

        return response()->json([
            'message' => 'Compte créé avec succès',
            'id_client' => $id
        ], 201);
    }

    public function login(Request $request)
    {
        $request->validate([
            'email' => 'required|email',
            'password' => 'required|string'
        ]);

        $client = DB::table('clients')->where('email', $request->email)->first();

        if (!$client || !Hash::check($request->password, $client->password)) {
            return response()->json(['message' => 'Identifiants invalides'], 401);
        }

        return response()->json([
            'message' => 'Connexion réussie',
            'id_client' => $client->id,
            'role' => $client->role,
            'email' => $client->email
        ]);
    }
}
