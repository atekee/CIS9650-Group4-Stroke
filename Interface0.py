print("Enter your own data to test the model: ")
age = int(input("Age:"))
work_type_Never_worked = input("Have you never worked?: （Yes/No)")
heart_disease_1 = input("Do you have any heart disease?: (Yes/No)")
avg_glucose_level = float(input("""What is your avg glucose level?:
(Range between..., if you don't know, type...)"""))
 # ... are the parts to be filled
ever_married_Yes = input("Have you ever married?:(Yes/No)")
hypertension_1 = input("Do you have hypertension?:(Yes/No)")
work_type_Private = input("Do you work in private sector?:(Yes/No)")

if work_type_Never_worked == "Yes":
    work_type_Never_worked = 1
elif work_type_Never_worked == "No":
    work_type_Never_worked = 0
    
if heart_disease_1 == "Yes":
    heart_disease_1 = 1
elif heart_disease_1 == "No":
    heart_disease_1 = 0
    
if ever_married_Yes == "Yes":
    ever_married_Yes = 1
elif ever_married_Yes == "No":
    ever_married_Yes = 0
    
if hypertension_1 == "Yes":
    hypertension_1 = 1
elif hypertension_1 == "No":
    hypertension_1 = 0

if work_type_Private == "Yes":
    work_type_Private = 1
elif work_type_Private == "No":
    work_type_Private = 0
    

