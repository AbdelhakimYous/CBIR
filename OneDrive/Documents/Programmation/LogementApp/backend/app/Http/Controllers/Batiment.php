<?php

namespace App\Http\Controllers;

use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Redirect;

class Batiment extends Controller
{
    public function createBatiment(Request $request)
    {
        $validated = $request->validate([
            'name' => ['required', 'max:255']
        ]);
        var_dump($validated);
        print_r($validated);

    }
}