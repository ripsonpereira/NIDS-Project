function OnShowData()
    {
        document.Form.action = "data"
        // document.Form.target = "_blank";    // Open in a new window
        document.Form.submit();             // Submit the page
        return true;
    }
    
    function OnPredict()
    {
        document.Form.action = "predict"
        // document.Form.target = "_blank";    // Open in a new window
        document.Form.submit();             // Submit the page
        return true;
    }