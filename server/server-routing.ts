let express = require('express');
var $ = require('jquery');
let ObjectID = require('mongodb').ObjectID;
let nodemailer = require('nodemailer');
const { google } = require("googleapis");
const OAuth2 = google.auth.OAuth2;

export class Server {
    private db;
    private server = express();
    private port = process.env.PORT;
    
    private router = express.Router();

    // Leave out database part for now
    constructor(db) {
        this.db = db;
        this.router.use((request, response, next) => {
            response.header('Content-Type','application/json');
            response.header('Access-Control-Allow-Origin', "*");
            response.header('Access-Control-Allow-Headers', '*');
        next();
        });

        // Serve static pages from a particular path.
        this.server.use('/', express.static('./static'));
        // Handle POST data as JSON
        this.server.use(express.json());
        this.server.use('/', this.router);

        // SEARCH
        this.server.post('/searchBook/', this.searchBookHandler.bind(this));
        this.server.get('/search/', function(req, res) {
            res.type('.html');
            res.sendFile('searchBook.html', { root: "../static" });
        });
        this.server.get('/searchResults/', function(req,res) {
            res.type('.html');
            res.sendFile('searchResults.html', { root: "../static" });
        });
        this.router.post('/postMessage/', this.postMessageHandler.bind(this));

        this.router.post('/login/', this.loginHandler.bind(this));
        this.router.post('/registerUser/', this.registerUser.bind(this));
        this.server.get('/options/', function(req, res) {
            res.type('.html');
            res.sendFile('selectActionAfterLogin.html', { root: "../static" });
        });
        this.server.get('/createAccount/', function(req, res) {
            res.type('.html');
            res.sendFile('createAccount.html', { root: "../static" });
        });
        this.server.get('/messages', function(req, res) {
            res.type('.html');
            res.sendFile('messages.html', { root: "../static" });
        });
        
        this.server.get('/sell/', function(req, res) {
            res.type('.html');
            res.sendFile('sellBook.html', { root: "../static" });
        });

        this.server.get('/rate/', function(req, res) {
            res.type('.html');
            res.sendFile('findUserToRate.html', { root: "../static" });
        });
        
        this.router.post('/accountInfo/', this.accountInfoHandler.bind(this));
        this.server.get('/accountInfo/', function(req, res){
            res.type('.html');
            res.sendFile('accountInfo.html', { root: "../static"});
        });
        this.server.get('/loadLogin/', function(req, res){
            res.type('.html');
            res.sendFile('login.html', { root: "../static"});
        });
        this.router.post('/checkNewAccount/', this.checkNewAccount.bind(this));
        this.router.get('/verifyAccount/', function(req, res) {
            res.type('html');
            res.sendFile('verifyOTP.html', {root: "../static"});
        });

        //router for checking your own postings
        this.server.get('/MyPostings/', function(req, res){
            res.type('html');
            res.sendFile('myPostings.html', {root: "../static"});
        });

        this.server.get('/rateUser/', function(req, res){
            res.type('.html');
            res.sendFile('userRating.html', {root: "../static"});
        });
        
    }   

    private getServer() {
        return this.server;
    }

    private async postMessageHandler(request, response) : Promise<void> {
        let user = request.body.user;
        let message = request.body.message;

        /*
            put data in server
        */
       response.write(JSON.stringify({'result':'success'}));
       response.end();
    }

    private async searchBookHandler(request,response) : Promise<void> {
        let query = request.body.query;
        const res = await this.db.getMany(
            {
                $or: [
                    { "title" : query},
                    { "author": query},
                    { "isbn": query}
                  ]
            }
            
           ,'bookPostings');
        if (res == null || res.length == 0) {
            response.write(JSON.stringify({
                'result': "nobooks",

            }));
        } else {
            response.write(JSON.stringify({
                'result': "success",
                'searchResults': res
            }));
        }

        response.end();
    }

    private async searchResultsHandler(request, response) : Promise<void> {
        response.write(JSON.stringify({"result": "success"}));
        response.end();
    }

    private async registerUser(request, response) : Promise<void> {
        // This email is unused and valid. Create a document in userInfo collection for this new user.
        let buyerRating = 0.0;
        let sellerRating = 0.0;
        let numBuyerRatings = 0;
        let numSellerRatings = 0;
        let queryObj = {
            'name': request.body.fullname,
            'email': request.body.email,
            'password': request.body.password,
            'institution': request.body.institution,
            'username': request.body.username,
            'buyerRating': 0.0,
            'sellerRating': 0.0,
            'numBuyerRatings': 0,
            'numSellerRatings': 0
        };
        const result = await this.db.putOne(queryObj, 'userInfo');
        response.write(JSON.stringify({
            'result': result
        }));
        response.end();
    }

    private async userRatingHandler(request, response) : Promise<void> {
        let userRating = request.body.rating;
        let type = request.body.rating;
        let user = request.body.username;

        response.write(JSON.stringify({
            'result': 'success'
        }));
        response.end();
    }

    private async loginHandler(request, response) : Promise<void> {
        var username = request.body.username;
        var password = request.body.password;
        // Now we find a document in userInfo collection that matches the above two.

        const res = await this.db.get({
            $and: [
                   { "username" : username},
                   { "password": password}
                 ]
          }, 'userInfo');
        var returnString = 'success';
        if(res == null) {
            returnString = 'failure';
        }
        response.write(JSON.stringify({
                "result": returnString }));
        response.end();
        return;
    }

    // dummy handler for the Account Info page
    private async accountInfoHandler(request, response) : Promise<void> {
        var username = request.body.username;
        var info = await this.db.get({"username": username}, 'userInfo');
        console.log(info);
        if( info == null){
            response.write(JSON.stringify({
                "result": "failure",
            }));
            response.end();
            return;
        }
        response.write(JSON.stringify({
            "result": "success",
            "username": username,
            "fullName": info.name,
            "institution": info.institution,
            "sRating": info.sellerRating,
            "bRating": info.buyerRating }));
        response.end();
    }

    // dummy handler for checking if account exists
    private async checkNewAccount(request, response) : Promise<void> {
        var email = request.body.email;
        var username = request.body.username;
        var fullName = request.body.fullName;
        // Check if email exists in db. If it does, return failure. 
        const result = await this.db.get(   {
            $or: [
                   { "email" : email },
                   { "username": username}
                 ]
          }, 'userInfo');
        if(result != null) {
            response.write(JSON.stringify({
                'result': 'failure'
            }));
            response.end();
            return;
        }

        //If it does not, send a 6-digit OTP to this email and return the OTP and success to the client
        var OTP = Math.floor(Math.random() * (999999 - 100000 + 1)) + 100000;
        let clientId, clientSecret, refreshToken;
        if(!process.env.CLIENTID) {
            let secrets = require('./secrets.json');
            clientId = secrets.clientId;
            clientSecret = secrets.clientSecret;
            refreshToken = secrets.refreshToken;
        } else {
            clientId = process.env.CLIENTID;
            clientSecret = process.env.CLIENTSECRET;
            refreshToken = process.env.REFRESHTOKEN;
        }
        // Send this OTP to the user for verification via email. 
        const oauth2Client = new OAuth2(
            clientId,
            clientSecret,
            "https://developers.google.com/oauthplayground"
       );
       oauth2Client.setCredentials({
           refresh_token: refreshToken
       });
       const accessToken = oauth2Client.getAccessToken();
       var transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                    type: "OAuth2",
                    user:'lakshayarora3107@gmail.com',
                    clientId: clientId,
                    clientSecret: clientSecret,
                    refreshToken: refreshToken,
                    accessToken: accessToken
                }
        });

        var mailOptions = {
            from: 'lakshayarora3107@gmail.com',
            to: email,
            subject: 'Passage OTP Verification',
            text: 'Hi ' + fullName + '. Welcome to Passage! Enter this OTP for account verification: ' + OTP
        };
        transporter.sendMail(mailOptions, function(error, info) {
            if(error) {
                console.log(error);
            } else {
                console.log('Email sent: ' + info.response);
            }
        });

        // Hard coded value returned for now
        response.write(JSON.stringify({
            'result': 'success',
            'OTP': OTP
        }));
        response.end();
    }

    public listen(port) : void {
        return this.server.listen(port);
    }

}