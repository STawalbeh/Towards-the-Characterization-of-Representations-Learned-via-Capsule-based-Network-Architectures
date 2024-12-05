import matplotlib.pyplot as plt

def trainingCurves(train, valid, test, label, title, saveImgPath):
  plt.figure(figsize=(10, 6))
  plt.plot(train, 'o-', color='green', label='Train-'+label)
  plt.plot(valid, 'o-', color='blue', label='Val-'+label)
  plt.plot(test, 'o-', color='pink', label='Test-'+label)
  
  plt.legend(prop={"size":18})
  plt.tick_params(axis='both', labelsize=19)
  plt.grid()
  plt.savefig('path/' + title +label +'.pdf', bbox_inches="tight")
