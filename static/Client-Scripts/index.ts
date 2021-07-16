const myURL = "http://localhost:8080/";

window.onload = function () {
    (async () => {
        getPapers();
    })();
}

async function registerInteraction() : Promise<void> {
    (async () => {
        //var user_id = sessionStorage.getItem('currentUser')
        // if(username == null){
        //     alert("Please Log In!");
        //     location.replace(myURL);
        // }

    })();
}

async function getPapers() {
    
}

async function postData(url : string, data: any) {
    const resp = await fetch(url,
    {
        method: 'POST',
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'same-origin',
        headers : { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        redirect: 'follow',
        body: JSON.stringify(data)
    });
    return resp; 
}