import spacy
import re

class EntityRecognizer:
    def __init__(self, model="en_core_web_lg"):
        """Initialize the NER model with larger model for better accuracy."""
        try:
            self.nlp = spacy.load(model)
        except:
            print(f"Model {model} not found. Falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def preprocess_text(self, text):
        """Clean and normalize text for better entity recognition."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix common issues
        text = text.replace('\n', ' ')
        return text
    
    def extract_entities(self, text):
        """Extract named entities with confidence scores."""
        text = self.preprocess_text(text)
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            # Calculate confidence based on context
            confidence = self._calculate_confidence(ent, doc)
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': confidence
            })
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlaps(entities)
        return entities
    
    def _calculate_confidence(self, ent, doc):
        """Calculate confidence score for entity."""
        # Base confidence from model
        confidence = 0.7
        
        # Boost for proper capitalization
        if ent.text[0].isupper():
            confidence += 0.1
        
        # Boost for multi-word entities
        if len(ent.text.split()) > 1:
            confidence += 0.1
        
        # Boost for known patterns
        if ent.label_ in ['DATE', 'MONEY', 'PERCENT']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _remove_overlaps(self, entities):
        """Remove overlapping entities, keeping higher confidence ones."""
        if not entities:
            return entities
        
        # Sort by start position
        sorted_ents = sorted(entities, key=lambda x: (x['start'], -x['confidence']))
        
        filtered = []
        for ent in sorted_ents:
            # Check if overlaps with any already added
            overlap = False
            for existing in filtered:
                if not (ent['end'] <= existing['start'] or ent['start'] >= existing['end']):
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(ent)
        
        return filtered
    
    def display_entities(self, text):
        """Display entities with their labels and confidence."""
        entities = self.extract_entities(text)
        print(f"\nText: {text}\n")
        print("Entities found:")
        for ent in entities:
            print(f"  - {ent['text']:20} -> {ent['label']:10} (confidence: {ent['confidence']:.2f})")
        return entities

def main():
    # Initialize recognizer
    recognizer = EntityRecognizer()
    
    # Sample texts
    sample_texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
        "The meeting is scheduled for next Monday at 3 PM in New York.",
        "Elon Musk's company Tesla is valued at over $800 billion."
    ]
    
    # Process each text
    for text in sample_texts:
        recognizer.display_entities(text)
        print("-" * 60)

if __name__ == "__main__":
    main()
