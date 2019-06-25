const tf = require('@tensorflow/tfjs-node')
const express = require("express");
const path = require("path");
const app = express();

app.use((req, res, next) => {
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next();
})

app.get('/predict', async (req, res) => {
    // const model = tf.sequential();
    // model.add(tf.layers.dense({ units: 1 }));
    // model.compile({
    //     loss: 'meanSquaredError',
    //     optimizer: 'sgd',
    //     metrics: ['MAE']
    // });


    const model = await tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'softmax' }),
        ]
    });
})

express.static(path.join(__dirname, "../static"));

app.listen(5000, () => {
    console.log("Serving on port 5000");
})