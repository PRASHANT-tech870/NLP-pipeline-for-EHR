package com.example.controller;

import com.example.hpcc.model;
import org.springframework.http.*;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@Controller
public class MyController {

    @GetMapping("/")
    public String index(Model model){
        // Initialize an empty model object to bind with the form
        model.addAttribute("user", new model());
        return "index";
    }

    @PostMapping("/sendJsonRequest")
    public String sendJsonRequest(@ModelAttribute model user, Model model) throws Exception{
        String requestBody = "{\"lets_try_once\":{\"med_data\":\"" + user.getMed_string() + "\"}}";
        System.out.println(user.toString());
        System.out.println(requestBody);
        // Configure HTTP headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Create HTTP entity with request body and headers
        HttpEntity<String> requestEntity = new HttpEntity<>(requestBody, headers);

        // Set the REST endpoint URL
        String url = "http://localhost:8002/WsEcl/json/query/roxie/lets_try_once"; // Replace with your actual API endpoint URL

        // Send the HTTP request
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> responseEntity = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);

        // Handle the response
        String responseBody = responseEntity.getBody();

        // Parse the JSON response body and extract data into lists
        List<String> symptomsList = new ArrayList<>();
        List<String> durationList = new ArrayList<>();
        List<String> genderList = new ArrayList<>();
        List<String> organList = new ArrayList<>();

        System.out.println(responseBody);
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode rootNode = objectMapper.readTree(responseBody);
            JsonNode resultsNode = rootNode.path("lets_try_onceResponse").path("Results").path("result_1").path("Row");

            for (JsonNode node : resultsNode) {
                symptomsList.add(node.path("symptoms").asText());
                durationList.add(node.path("duration_of_persistance").asText());
                genderList.add(node.path("gender").asText());
                organList.add(node.path("organ").asText());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        user.setL1(symptomsList);
        user.setL2(durationList);
        user.setL3(genderList);
        user.setL4(organList);

        // Add lists to model for use in the response template
        model.addAttribute("symptomsList", user.getL1());
        model.addAttribute("durationList", user.getL2());
        model.addAttribute("genderList", user.getL3());
        model.addAttribute("organList", user.getL4());

        System.out.println(user.getL1());
        System.out.println(user.getL2());
        System.out.println(user.getL3());
        System.out.println(user.getL4());




        // Return the name of the response template
        return "response";
    }
}
