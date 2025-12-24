# Definizione di una funzione di plotting:
#   def <nome_del_plot>(<args>):
#       <logica di manipolazione dei dati>
#       plt.close() # o equivalente per pulire la RAM
#
# Definizione di una funzione helper:
#   la funzione deve essere la più generica possibile,
#   astraendo i dettagli specifici del dataset. Se la logica richiede specificità,
#   forse è meglio inserirla in una funzione di plotting.
#   Devono restituire al massimo una figura senza visualizzarla o salvarla.
#   le funzioni helper iniziano con "_".
#
#   def _<nome_funzione_helper>(<args>) -> <figura>:
#       <logica helper>
#       return <figura>


