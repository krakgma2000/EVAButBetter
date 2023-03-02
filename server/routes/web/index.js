module.exports = (app) => {
  const express = require("express");
  const assert = require("http-assert");
  const jwt = require("jsonwebtoken");
  const axios = require("axios");
  const router = express.Router({
    mergeParams: true,
  });
  router.post("/", async (req, res) => {
    var data = req.body;
    content = {
      "content": data
    }
    try {
      axios.post("http://localhost:8000/get_msg", content).then((result) => {
        var data = result.data
        res.send(data);
        console.log(result);
      });
    } catch (error) {
      console.log(error);
    }


  });

  router.get("/:id", async (req, res) => {
    const id = req.params;
    console.log("receive request");
    res.send(id);
  });
  app.use("/web", router);

  app.use(async (err, req, res, next) => {
    // console.log(err)
    res.status(err.statusCode || 500).send({
      message: err.message,
    });
  });
};
