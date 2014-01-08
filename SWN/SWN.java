import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SWN {

  private Map<String, Double> dictionary;

  public SWN(String pathToSWN) throws IOException {
	
    // This is our main dictionary representation
    dictionary = new HashMap<String, Double>();

    // From String to list of doubles.
    HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

    BufferedReader csv = null;
    try {
      csv = new BufferedReader(new FileReader(pathToSWN));
      int lineNumber = 0;

      String line;
      while ((line = csv.readLine()) != null) {
	lineNumber++;

	// If it's a comment, skip this line.
	if (!line.trim().startsWith("#")) {
	  // We use tab separation
	  String[] data = line.split("\t");
	  String wordTypeMarker = data[0];

	  
	  if (data.length != 6) {
	    throw new IllegalArgumentException(
					       "Incorrect tabulation format in file, line: "
					       + lineNumber);
	  }

	  // Calculate synset score as score = PosS - NegS
	  Double synsetScore = Double.parseDouble(data[2])
	    - Double.parseDouble(data[3]);

	  // Get all Synset terms
	  String[] synTermsSplit = data[4].split(" ");

	  // Go through all terms of current synset.
	  for (String synTermSplit : synTermsSplit) {
	    // Get synterm and synterm rank
	    String[] synTermAndRank = synTermSplit.split("#");
	    String synTerm = synTermAndRank[0] + "#"
	      + wordTypeMarker;

	    int synTermRank = Integer.parseInt(synTermAndRank[1]);
	    // What we get here is a map of the type:
	    // term -> {score of synset#1, score of synset#2...}

	    // Add map to term if it doesn't have one
	    if (!tempDictionary.containsKey(synTerm)) {
	      tempDictionary.put(synTerm,
				 new HashMap<Integer, Double>());
	    }

	    // Add synset link to synterm
	    tempDictionary.get(synTerm).put(synTermRank,
					    synsetScore);
	  }
	}
      }


      for (Map.Entry<String, HashMap<Integer, Double>> entry : tempDictionary
	     .entrySet()) {
	String word = entry.getKey();
	Map<Integer, Double> synSetScoreMap = entry.getValue();

	double score = 0.0;
	double sum = 0.0;
	for (Map.Entry<Integer, Double> setScore : synSetScoreMap
	       .entrySet()) {
	  score += setScore.getValue() / (double) setScore.getKey();
	  sum += 1.0 / (double) setScore.getKey();
	}
	score /= sum;

	dictionary.put(word, score);
      }
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      if (csv != null) {
	csv.close();
      }
    }
  }

  public String extract(String word, String pos) {
	 
    double value= 0;
   
    value = dictionary.get(word + "#" + pos)==null?0:dictionary.get(word + "#" + pos);
    String feeling=null;
    
    if(value<=1.0&&value>0.75){
    	feeling = "strong-positive";
    } else if(value<=0.75&&value>0.5){
    	feeling = "positive";
    }else if(value<=0.5&&value>0.25){
    	feeling = "weak-positive";
    }else{
    	feeling = "negative";
    }
    return feeling;
  }


public String extractSemantics(String word, String pos) {
	 
    double value= 0;
   
    value = dictionary.get(word + "#" + pos)==null?0:dictionary.get(word + "#" + pos);
    String feeling=null;
    
    if(Math.abs(value)>0.5){
    	return "subj";
    }else{
    	return "obj";
    }
   
  
}
public double extractValue(String word, String pos) {
	 
    double value= 0;
   
    value = dictionary.get(word + "#" + pos)==null?0:dictionary.get(word + "#" + pos);
    String feeling=null;
    
    return value;
  }
}
