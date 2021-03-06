export class Database {
    private MongoClient = require('mongodb').MongoClient;
    private uri;
    private client;
    private db;

    constructor() {
        if(!process.env.MONGOURI) {
            let secrets = require('./secrets.json');
            this.uri = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false';
        }
        else {
            this.uri = process.env.MONGOURI;
        }
        this.client = new this.MongoClient(this.uri, { useNewUrlParser: true, useUnifiedTopology: true});
        // Open up a connection to the client
        (async () => {
            await this.client.connect().catch(err => { console.log(err); });
            this.db = this.client.db('papers');
            console.log('Successfully connected to the database');
        })();
    }


    public async get(query, collectionName: string) : Promise<any> {
        let collection = this.db.collection(collectionName);
        let result = await collection.findOne(query);
        return result;
    }

    public async putOne(query, collectionName: string) : Promise<string> {
        let collection = this.db.collection(collectionName);
        let result = await collection.insertOne(query).then(
            val => {
                return 'success';
            },
            reason => {
                return 'failure';
            }
        );
        return result;
    }

    public async updateSingular(query, newValues, collectionName: string): Promise<string> {
        let collection = this.db.collection(collectionName);
        let result = await collection.updateOne(query, newValues);
        if(result.modifiedCount == 1)
            return 'success';
        else
            return 'failure';
    }
    
    public async getMany(query, collectionName: string) : Promise<string> {
        let collection = this.db.collection(collectionName);
        console.log(query);
        let res = await collection.find(query).toArray().then(
            data => {
                console.log(data);
                return data;
            },
            err => {
                console.log(err);
                return null;
            }
        );
        return res;
    }

    public async delete(query, collectionName: string): Promise<string> {
        let collection = this.db.collection(collectionName);
        console.log(query);
        let res = await collection.deleteMany(query)
            .then(result => {
                console.log(result);
                return result;
            })
            .catch( err => {
                console.log(err);
                return null;
            });
        return res;
    }
}   