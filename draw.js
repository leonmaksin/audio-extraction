const io = require('socket.io-client');
const socket = io.connect('http://localhost:5001');

socket.on('connect', () => {
    console.log('Connected to server');
    socket.emit('ready');
});

socket.on('data', (data) => {
    console.log('Received data:', data);
    socket.emit('received');
});
