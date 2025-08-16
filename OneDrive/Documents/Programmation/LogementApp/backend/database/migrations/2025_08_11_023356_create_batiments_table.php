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
        Schema::create('batiments', function (Blueprint $table) {
            $table->id();
            $table->foreignId('id_client')->constrained('clients')->onDelete('cascade'); // FK vers batiments
            $table->string('nom');
            $table->string('type')->nullable();
            $table->string('adresse');
            $table->string('ville');
            $table->string('province');
            $table->string('code_postal'); 
            $table->text('description')->nullable();
            $table->decimal('prix_moyen', 10, 2)->nullable();
            $table->decimal('latitude', 10, 7);
            $table->decimal('longitude', 10, 7); 
            $table->date('date_debut_disponibilite')->nullable(); // date début de dispo
            $table->date('date_fin_disponibilite')->nullable();   // date fin de dispo
            });
}

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('batiments');
    }
};
