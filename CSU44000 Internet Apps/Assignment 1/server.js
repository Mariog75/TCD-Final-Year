var express = require("express");
var path = require("path");
var bodyParser = require("body-parser");
const fs = require('fs');
const readline = require('readline');
const unirest = require("unirest");
const request = require('request');

var app = express();
app.use(express.static(__dirname + "/public"));
app.use(bodyParser.json());

var allowCrossDomain = function(req, res, next) {
    res.header('Access-Control-Allow-Origin', "*");
    res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
}

app.use(allowCrossDomain);

var server = app.listen(process.env.PORT || 8080, function () {
    var port = server.address().port;
    console.log("App now running on port", port);
});
function getWeather(location, callback) {
    var options = {
        method: 'GET',
        url: 'https://community-open-weather-map.p.rapidapi.com/forecast',
        qs: {q: location},
        headers: {
          'x-rapidapi-host': 'community-open-weather-map.p.rapidapi.com',
          'x-rapidapi-key': 'b746d544c0mshe2f209d75f6f736p1e2735jsn897a8a83965f'
        }
      };
    
    request(options, function(err, res, body) {
    if (err) {
        return callback(null, err)
    }
    console.log(body);
    return callback(body, false);
    });
}

//api GET
app.get("/api", function (req, res) {
    res.sendStatus(200);
});
//weather GET
app.get("/api/weather", function (req, res) {
    
    var location = req.param('location');

    getWeather(location, function(err, data){ 
        if(err) return res.send(err);   

        var list = data.list;

        res.send(list);

    });

});