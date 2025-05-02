# DO/BE/FEEL Motivational Model  

## **Who (Stakeholders)**  
_List the primary stakeholders interacting with your system (e.g., users, administrators, developers)._  
- The Research Scientist
- Student
- Software Engineers
- System Administrators


 # System Goals

## Functional Goals
*What the system should do and accomplish*

### Visualization & Analysis
- **End-to-End Workflow Visualization**: Display each stage of the model/package pipeline (pre-processing, prelim, sifted, pilot, cloister, Pythia, trace)
- **Instance Space Exploration**: Provide interactive plots to analyze distributions and performance trends
- **Model Comparison**: Allow users to compare multiple experiment runs side by side
- **Custom Metrics & Filters**: Enable filtering and sorting of results based on user-defined parameters
- **Algorithm Performance Visualization**: Display performance metrics for all algorithms in comparative views
- **Footprint Visualization**: Show algorithm footprints and regions where specific algorithms excel
- **Cross-validation Results**: Visualize prediction accuracy, precision, and other validation metrics

### Integration & Data Handling
- **Export & Sharing**: Allow users to export visualizations and reports (CSV, PNG, PDF, SVG, Excel)
- **Result Caching**: Store frequently accessed experiment results to reduce redundant computations
- **Batch Processing**: Support processing multiple datasets efficiently
- **Data Selection**: Enable filtering and selecting specific instances for focused analysis

### Security & Access Control
- **User Authentication**: Login-based access with strong password credentials
- **Secure API Endpoints**: Protect data exchange between backend and frontend with authentication tokens
- **Session Management**: Implement appropriate timeout procedures

## Non-Functional Goals
*How the system should perform its functions*

### Performance
- **Speed**: 
  - Visualizations must load within 3 seconds with standard datasets
  - Frequently accessed data must load in under 1 second
  - Initial page load under 5 seconds on standard connections
  - Visualization updates within 2 seconds when changing displayed algorithms
  - Hover tooltips must appear within 200ms

- **Scalability**: 
  - Support 200-500 simultaneous users with load times of 2-5 seconds
  - Handle batch processing of multiple datasets without UI blocking

### Usability
- **Interface Design**: 
  - Intuitive parameter controls with appropriate limits and step sizes
  - Clear toggle controls for algorithm visualization
  - Responsive design adapting to different screen sizes

- **Documentation**: 
  - Contextual help for each visualization type
  - Step-by-step tutorials covering all basic functionality
  - Searchable documentation

### Security
- **Data Protection**: 
  - Two-factor authentication for sensitive data
  - HTTPS encryption for all API requests
  - Input validation to prevent injection attacks

- **Access Control**:
  - Failed login attempt limitations and logging
  - 30-minute session timeout for inactivity

### Reliability
- **Error Handling**: 
  - Clear error messages during system overload
  - Partial failures in batch processing should not stop entire batch

- **Browser Compatibility**: 
  - Function correctly in Chrome, Firefox, Safari, and Edge
  - Consistent performance across all supported browsers

### Maintainability
- **Caching Management**:
  - Intelligent caching based on usage patterns
  - Manual cache refresh capabilities
  - Cache status indicators showing when data was last updated


## **BE (Quality Goals)**  
_How should the system behave? Identify key quality attributes._  
- Secure  
- Accessible  
- Scalable  
- Easy to use
- Robust

## **FEEL (Emotional Goals - OPTIONAL)**  
_How should stakeholders feel when interacting with the system?_  
- Confident  
- Engaged  
- Supported
- Empowered

## **Motivational Model**  
_Based on the above lists, consider creating a simple Motivational Model diagram (e.g., using Miro, Draw.io) to represent the relationships between these elements._  
![](https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/blob/main/docs/Motivational_Model_diagram.jpeg)

