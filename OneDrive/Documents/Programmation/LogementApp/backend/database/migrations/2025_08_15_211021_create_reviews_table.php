<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('reviews', function (Blueprint $table) {

            $table->unsignedBigInteger('id_client_expediteur');
            $table->unsignedBigInteger('id_client_destinataire');
            $table->text('contenu');
            $table->timestamps();
            $table->foreign('id_client_expediteur')->references('id')->on('clients')->onDelete('cascade');
            $table->foreign('id_client_destinataire')->references('id')->on('clients')->onDelete('cascade');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('reviews');
    }
};

