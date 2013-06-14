import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(9, 1.5))
ax = fig.add_subplot(111)

x=np.arange(0, 10, 0.1)
y=2*np.tan(x)

ax.set_xlim(0, 10)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
#http://matplotlib.sourceforge.net/users/mathtext.html
ax.set_xticklabels([r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$",r"$10^{5}$",r"$10^{6}$",r"$10^{7}$",r"$10^{8}$",r"$10^{9}$" ])

ax.set_ylim(0, 6)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
#http://matplotlib.sourceforge.net/users/mathtext.html
ax.set_yticklabels([r"$\Delta$", r"$\Gamma$",r"$\Lambda$",r"$\Omega$",r"$\Phi$",r"$\Pi$",r"$\Psi$"])

#ax.text(1, 1, r'$\mathtt{matplotlib}\/{examples}$')
#ax.text(2, 2, r'$\mathscr{matplotlib}\/\mathcircled{examples}$', size=25)
#ax.bar(x,y, alpha=0.1, color="red")
#plt.subplots_adjust(left=0.08, bottom=0.33, right=0.96, top=0.83, wspace=0.05, hspace=0)
plt.show()
