export default function AddHouse (){
    async function getFormData(data:any)
    {
        'use server'
        console.log("ca marche")
        const name = data.get('name')
        console.log(name)
        await fetch("http://127.0.0.1:8000/api/creerBatiment",{
            method:"POST",
              headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({name:name})
        }) 

    }

    
    return(

        <form action={getFormData}>
            <input type="text" name="name"/>
            <button type="submit">submit</button>
        </form>
    )
}