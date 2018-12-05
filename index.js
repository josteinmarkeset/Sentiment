// Uses pre-trained google CNN's for maximum accuracy
const URLS = {
    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
}

const TWEET_COUNT = 100;
const POSITIVE_IMAGE_URL = 'img/positive.png';
const NEGATIVE_IMAGE_URL = 'img/negative.png';

const inputElement = document.getElementById('inputText');
const predictButton = document.getElementById('predict');
const positivityText = document.getElementById('positivityText');
const sentimentImage = document.getElementById('sentimentImage');
const positivityMeter = document.getElementById('positivityMeter');

predictButton.onclick = run;
// Allows "Enter" key to be pressed to trigger analysis
inputElement.addEventListener("keyup", event => {
    event.preventDefault();
    if (event.keyCode === 13) {
      predictButton.click();
    }
});

// To init vars
let model;
let metadata;
let indexFrom;
let maxLen;
let wordIndex;

let loaded = false;

var twitter = new Codebird;
twitter.setConsumerKey('ICZuo7AkmO0IbGVLu1TZ47h3T', 'A2rodRQLk3hJ1j0ST1YTtcdnN9BK9l46LNyZW4rQJQAWVReZVL');
init();

function run() {
    const search = inputElement.value.toString(); // toString() to make sure we actually get a string
    getRecentTweets(search, TWEET_COUNT);
}

// Load in async function as required by tensorflow
async function init() {
    // Load model and metadata
    model = await loadHostedPretrainedModel(URLS.model);
    metadata = await loadHostedMetadata(URLS.metadata);

    // Set vars
    indexFrom = metadata['index_from'];
    maxLen = metadata['max_len'];
    wordIndex = metadata['word_index'];

    predictButton.disabled = false;
    loaded = true;
}

function predict(inputText) {
    if(!loaded) return alert('Warning! No model is loaded.'); // Make sure model is loaded

    inputText = inputText.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' '); // Convert to lower case and remove all punctuations.

    // Convert words to indices
    const inputBuffer = tf.buffer([1, maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
        const word = inputText[i];
        inputBuffer.set(wordIndex[word] + indexFrom, 0, i);
    }

    const input = inputBuffer.toTensor();
    const beginMs = Date.now();
    const predictOut = model.predict(input);
    const positivity = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = Date.now();

    /*
    logResults({ 
        positivity: positivity, 
        elapsed: (endMs - beginMs)
    });*/

    return positivity;
}

function logResults(result) {
    const pPositivity = result.positivity * 100;
    console.log(`Positivity: ${pPositivity}%`);
    console.log(result.elapsed);
}

async function loadHostedPretrainedModel(url) {
    try {
        const model = await tf.loadModel(url);
        console.log('Successfully loaded model!');
        return model;
    } catch (err) {
        console.error(err);
        alert('Failed loading model!');
    }
}

async function loadHostedMetadata(url) {
    try {
        const metadataJson = await fetch(url);
        return await metadataJson.json();
    } catch (err) {
        console.error(err);
    }
}

function showResult(mean) {
    const pMean = Math.round(mean * 100);
    positivityMeter.innerText = pMean + '%';

    if(pMean >= 50)
        sentimentImage.src = POSITIVE_IMAGE_URL;
    else
        sentimentImage.src = NEGATIVE_IMAGE_URL;

    positivityText.style.display = 'block';
    sentimentImage.style.display = 'inline-block';
}

function analysePositivity(tweets) {
    let positivitySum = 0;
    for (let i = 0; i < tweets.length; i++) {
        positivitySum += predict(tweets[i].text);
    }

    const mean = positivitySum / tweets.length;
    showResult(mean);
}

function getRecentTweets(search, tweetCount = 15) {
    const params = {
        q: search,
        result_type: 'popular', // Filtering by "popular" will be more representative of popular opinion
        count: tweetCount.toString()
    }

    twitter.__call('search_tweets', params, reply => {
        if(reply.statuses.length < 1) return alert('No tweets found about ' + search);
        analysePositivity(reply.statuses);
    });
}

const calcValue = x => (x*2) + 3;