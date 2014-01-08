 
import java.io.*;

import net.didion.jwnl.data.POS;

public class SWNUtil {

        /**
         * String that stores the text to guess its polarity.
         */
         String text;
        
        /**
         * SentiWordNet object to query the polarity of a word.
         */
         SWN3 sentiwordnet =null;;                
        
         /**
         * This method loads the text to be classified.
         * @param fileName The name of the file that stores the text.
         * 
         */
        public SWNUtil(){
        	try {
				sentiwordnet = new SWN3("/home/sreejat/workspace/Copy of sentiAnalysis/src/sentiWord.txt");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
        public void load(String fileName) {
                try {
                        BufferedReader reader = new BufferedReader(new FileReader(fileName));
                        String line;
                        text = "";
                        while ((line = reader.readLine()) != null) {
                text = text + " " + line;
            }
                        // System.out.println("===== Loaded text data: " + fileName + " =====");
                        reader.close();
                        // System.out.println(text);
                }
                catch (IOException e) {
                        System.out.println("Problem found when reading: " + fileName);
                }
        }

        /**
         * This method performs the classification of the text.
         * Algorithm: Use all POS, say "yes" in case of 0.
         * @return An string with "no" (negative) or "yes" (positive).
         */
        public  String classifyAllPOSY(String phrase,String mode) {
        
                int count =0;
                String feeling = null;
                                // Add weights -- positive => +1, strong_positive => +2, negative => -1, strong_negative => -2
                                if (phrase!=null&&!phrase.equals("")) {
                                        // Search as adjetive
                                        feeling = sentiwordnet.extract(phrase,mode);
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                                if(feeling.equals("strong_positive")){
                                                	count += 2;
                                                }else if(feeling.equals("positive")){
                                                	count+=1;
                                                }else if(feeling.equals("negative")){
                                                	count-=1;
                                                }else if(feeling.equals("strong_negative")){
                                                	count-=2;
                                                }
                                                       
                                                }
                                                // System.out.println(tokens[i]+"#"+feeling+"#"+count);
                                        }
                                        
                        // System.out.println(count);
                
                // Returns "yes" in case of 0
                if (count >= 0)
                        return "yes";
                else return "no";
        }
        
        /**
         * This method performs the classification of the text.
         * Algorithm: Use all POS, say "no" in case of 0.
         * @return An string with "no" (negative) or "yes" (positive).
         */
        public String classifyAllPOSN() {
        
                int count = 0;
                try {
                        String delimiters = "\\W";
                        String[] tokens = text.split(delimiters);
                        String feeling = "";
                        for (int i = 0; i < tokens.length; ++i) {
                                // Add weights -- positive => +1, strong_positive => +2, negative => -1, strong_negative => -2
                                if (!tokens[i].equals("")) {
                                        // Search as adjetive
                                        feeling = sentiwordnet.extract(tokens[i],"a");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                         
                                                           
                                                    }
                                        // Search as noun
                                        feeling = sentiwordnet.extract(tokens[i],"n");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                            }
                                        // Search as adverb
                                        feeling = sentiwordnet.extract(tokens[i],"r");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                            }
                                        // Search as verb
                                        feeling = sentiwordnet.extract(tokens[i],"v");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                            }
                                }
                        }
                        // System.out.println(count);
                }
                catch (Exception e) {
                        System.out.println("Problem found when classifying the text"+e);
                }
                // Returns "no" in case of 0
                if (count > 0)
                        return "yes";
                else return "no";
        }
        
        /**
         * This method performs the classification of the text.
         * Algorithm: Use only ADJ, say "yes" in case of 0.
         * @return An string with "no" (negative) or "yes" (positive).
         */
        public String classifyADJY() {
        
                int count = 0;
                try {
                        String delimiters = "\\W";
                        String[] tokens = text.split(delimiters);
                        String feeling = "";
                        for (int i = 0; i < tokens.length; ++i) {
                                // Add weights -- positive => +1, strong_positive => +2, negative => -1, strong_negative => -2
                                if (!tokens[i].equals("")) {
                                        // Search as adjetive
                                        feeling = sentiwordnet.extract(tokens[i],"a");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                            } 
                                }
                        }
                        // System.out.println(count);
                }
                catch (Exception e) {
                        System.out.println("Problem found when classifying the text"+e);
                }
                // Returns "yes" in case of 0
                if (count >= 0)
                        return "yes";
                else return "no";
        }
        
        /**
         * This method performs the classification of the text.
         * Algorithm: Use only ADJ, say "no" in case of 0.
         * @return An string with "no" (negative) or "yes" (positive).
         */
        public String classifyADJN() {
        
                int count = 0;
                try {
                        String delimiters = "\\W";
                        String[] tokens = text.split(delimiters);
                        String feeling = "";
                        for (int i = 0; i < tokens.length; ++i) {
                                // Add weights -- positive => +1, strong_positive => +2, negative => -1, strong_negative => -2
                                if (!tokens[i].equals("")) {
                                        // Search as adjetive
                                        feeling = sentiwordnet.extract(tokens[i],"a");
                                        if ((feeling != null) && (!feeling.equals(""))) {
                                            if(feeling.equals("strong_positive")){
                                            	count += 2;
                                            }else if(feeling.equals("positive")){
                                            	count+=1;
                                            }else if(feeling.equals("negative")){
                                            	count-=1;
                                            }else if(feeling.equals("strong_negative")){
                                            	count-=2;
                                            }
                                                   
                                            }
                                }
                        }
                        // System.out.println(count);
                }
                catch (Exception e) {
                        System.out.println("Problem found when classifying the text");
                }
                // Returns "no" in case of 0
                if (count > 0)
                        return "yes";
                else return "no";
        }
 
        /**
         * Main method.
         * Usage: java SentiWordNetDemo <file>
         * @param args The command line args.
         */
        public String extractSentiment(String phrase,String POS){
           	text = phrase;
        	String mode = "n";
        	//System.out.println(phrase+" phrase");
        	String feeling = null;
        	if(POS.startsWith("VB")){
				mode = "v";
			}else if(POS.startsWith("NN")){
				mode = "n";
			}else if(POS.startsWith("JJ")){
				mode = "a";
			}else if(POS.startsWith("RB")){
				mode = "r";
			}
        	String temp =classifyAllPOSY(phrase,mode);
        	if(temp=="yes"){
        		feeling = "positive";
        	}else{
        		feeling = "negative";
        	}
        	return feeling;
        	
        }
        public String extractSemantic(String phrase,String POS){
        return sentiwordnet.extractSemantics(phrase,POS);
        }
        public double extractSentimentValue(String phrase,String POS){
        	return sentiwordnet.extractValue(phrase,POS);
        }
        
}
