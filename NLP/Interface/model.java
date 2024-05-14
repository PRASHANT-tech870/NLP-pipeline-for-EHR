package com.example.hpcc;

import java.util.List;

public class model {

    private String med_string;
    private List<String> l1;
    private List<String> l2;
    private List<String> l3;

    public List<String> getL1() {
        return l1;
    }

    public void setL1(List<String> l1) {
        this.l1 = l1;
    }

    public List<String> getL2() {
        return l2;
    }

    public void setL2(List<String> l2) {
        this.l2 = l2;
    }

    public List<String> getL3() {
        return l3;
    }

    public void setL3(List<String> l3) {
        this.l3 = l3;
    }

    public List<String> getL4() {
        return l4;
    }

    public void setL4(List<String> l4) {
        this.l4 = l4;
    }

    private List<String> l4;
    public String getMed_string() {
        return med_string;
    }

    public void setMed_string(String med_string) {
        this.med_string = med_string;
    }

    // Other fields and methods...



    @Override
    public String toString() {
        return "model{" +
                "med_string=" + med_string+

                "}";
    }
}