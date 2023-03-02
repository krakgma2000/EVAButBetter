const express = require("express")

const app = express()


app.use(require('cors')())
app.use(express.json())

require('./routes/web')(app)

app.listen(3000, () => {
  console.log('http://localhost:3000');
});