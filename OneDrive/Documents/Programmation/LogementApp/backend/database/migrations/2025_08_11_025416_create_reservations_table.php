<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
public function up(): void
{
    Schema::create('reservations', function (Blueprint $table) {
        $table->id();
        $table->foreignId('id_client')->constrained('clients')->onDelete('cascade'); // FK vers table clients
        $table->foreignId('id_batiment')->constrained('batiments')->onDelete('cascade'); // FK vers batiments
        $table->date('date_debut'); // uniquement année-mois-jour
        $table->date('date_fin');
        $table->float('prix_moyen');
           // uniquement année-mois-jour
    });
}


    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('reservations');
    }
};
