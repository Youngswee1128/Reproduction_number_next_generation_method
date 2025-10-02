import sympy as sp
from sympy import symbols,  simplify, latex, pprint
from typing import List, Dict,  Union

#下面是定义的智能化基本再生数计算器类
class SmartR0Calculator:
    """
    智能化基本再生数符号计算器
    
    能够根据传染病模型的微分方程自动提取F和V矩阵，并计算基本再生数R₀
    
    使用方法：
    1. 定义模型的微分方程组
    2. 指定感染相关仓室
    3. 自动提取F矩阵和V矩阵
    4. 计算基本再生数R₀
    """
    #这个是初始化函数 self是类的实例
    def __init__(self):                          
        self.model_equations = {}              # 存储模型方程，为字典类型
        self.infected_compartments = []        # 感染相关仓室，仓室的数据类型为list列表
        self.all_compartments = []             # 所有仓室，仓室的数据类型为list列表
        self.parameters = set()                # 模型参数，这里参数类型为set集合
        self.F_matrix = None                   # 传播矩阵，传播矩阵的数据类型为Matrix矩阵
        self.V_matrix = None                   # 转移矩阵，转移矩阵的数据类型为Matrix矩阵
        self.NGM = None                        # 下一代矩阵，下一代矩阵的数据类型为Matrix矩阵
        self.R0 = None                         # 基本再生数，基本再生数的数据类型为符号表达式
        self.disease_free_equilibrium = {}     # 无病平衡点，无病平衡点的数据类型为字典

    #这个函数是定义模型的微分方程组
    def infectious_model(self, equations: Dict[str, sp.Expr], infected_compartments: List[str]):
        
        """
        定义传染病模型
        参数:
        equations (dict): 微分方程组，格式为 {'S': dS/dt表达式, 'I': dI/dt表达式, ...}
        infected_compartments (list): 感染相关仓室列表，例如 ['E', 'I']
        
        示例：
        equations = {
            'S': -beta * S * I / N,
            'I': beta * S * I / N - gamma * I,
            'R': gamma * I
        }
        infected_compartments = ['I']
        """
        print("=" * 60)
        print("正在定义传染病模型...")
        
        self.model_equations = equations                      # 获取存储模型方程
        self.infected_compartments = infected_compartments    # 获取感染相关仓室
        self.all_compartments = list(equations.keys())        # 获取所有仓室
        
        # 提取所有参数
        for eq in equations.values():
            self.parameters.update(eq.free_symbols)
        
        # 移除仓室变量，只保留参数
        compartment_symbols = {sp.Symbol(comp) for comp in self.all_compartments}
        self.parameters = self.parameters - compartment_symbols
        
        
        # 显示模型信息 有哪些仓室 感染仓室 参数
        print(f"仓室: {self.all_compartments}")
        print(f"感染仓室: {self.infected_compartments}")
        print(f"参数: {sorted([str(p) for p in self.parameters])}")
        
        # 显示模型方程
        print("\n模型方程组:")
        for comp, eq in equations.items():
            print(f"d{comp}/dt = {eq}")
        
        return self
    
    #这个函数是设置无病平衡点,分为两个部分，
    #如果没有提供平衡点，则自动假设所有感染仓室为0，易感仓室为符号变量 例如 S0
    #如果提供了平衡点，则直接使用用户提供的平衡点 
    def set_disease_free_equilibrium(self, equilibrium: Dict[str, sp.Expr] = None):
        """
        设置无病平衡点
        
        参数:
        equilibrium (dict): 无病平衡点，格式为 {'S': S0, 'I': 0, 'R': 0, ...}
                           如果不提供，将自动假设所有感染仓室为0
        """
        if equilibrium is None:
            # 自动设置：感染仓室为0，易感仓室为总人口
            self.disease_free_equilibrium = {}
            for comp in self.all_compartments:
                if comp in self.infected_compartments:
                    self.disease_free_equilibrium[comp] = 0
                else:
                    # 对于非感染仓室，需要用户指定或假设
                    self.disease_free_equilibrium[comp] = sp.Symbol(f"{comp}0")
        else:
            self.disease_free_equilibrium = equilibrium
        
        print("\n无病平衡点:")
        for comp, val in self.disease_free_equilibrium.items():
            print(f"{comp}* = {val}")
        
        return self
    
    #这个函数是提取F矩阵和V矩阵 里面调用了两个子函数
    #调用了_construct_F_matrix和_construct_V_matrix
    #最后返回的是 F矩阵和V矩阵
    def extract_F_V_matrices(self):
        """
        自动提取F矩阵和V矩阵
        
        F矩阵: 新感染项 (从易感仓室到感染仓室的流入)
        V矩阵: 感染仓室间的转移和流出项
        
        对于SEIR模型：
        - F矩阵应该只包含新感染项
        - V矩阵应该包含E和I之间的转移以及各自的流出
        """
        print("\n" + "=" * 60)
        print("正在提取F矩阵和V矩阵...")
        # 初始化F和V矩阵
        n_infected = len(self.infected_compartments)
        F_matrix = sp.zeros(n_infected, n_infected)
        V_matrix = sp.zeros(n_infected, n_infected)
        
        # 首先构造V矩阵 - 分析感染仓室之间的转移关系
        self._construct_V_matrix(V_matrix)
        
        # 然后构造F矩阵 - 分析新感染项
        self._construct_F_matrix(F_matrix)
        
        self.F_matrix = simplify(F_matrix)
        self.V_matrix = simplify(V_matrix)
        
        print(f"\n提取结果:")
        print("F矩阵 (新感染项):")
        pprint(self.F_matrix)
        print("\nV矩阵 (转移项):")
        pprint(self.V_matrix)
        
        return self.F_matrix, self.V_matrix
    
    #这个函数是构造V矩阵
    def _construct_V_matrix(self, V_matrix):
        """
        构造V矩阵：感染仓室的转移和流出
        """
        print("\n构造V矩阵...")
        n_infected = len(self.infected_compartments)
        
        # 为每个感染仓室分析其流入和流出
        for i, comp_i in enumerate(self.infected_compartments):
            equation = self.model_equations[comp_i]
            comp_i_symbol = sp.Symbol(comp_i)
            
            print(f"\n分析仓室 {comp_i}: d{comp_i}/dt = {equation}")
            
            # 分析每一项
            terms = sp.Add.make_args(equation)
            
            # 计算该仓室的净流出率（对角元素）
            total_outflow = 0
            
            for term in terms:
                # 跳过新感染项（包含易感仓室的项）
                if self._contains_susceptible_interaction(term):
                    continue
                
                # 检查是否为该仓室的流出项
                if comp_i_symbol in term.free_symbols:
                    # 线性化并提取系数
                    linearized_term = self._linearize_term_at_dfe(term)
                    coeff = linearized_term.coeff(comp_i_symbol, 1)
                    if coeff is not None and coeff != 0:
                        total_outflow += -coeff  # 流出项是负的，所以取负号
                        print(f"  流出项: {term} -> 系数: {coeff} -> 累计流出: {total_outflow}")
            
            # 设置对角元素
            V_matrix[i, i] = total_outflow
            print(f"  V[{i},{i}] = {total_outflow}")
            
            # 检查从其他感染仓室的流入（非对角元素）
            for j, comp_j in enumerate(self.infected_compartments):
                if i == j:
                    continue
                
                comp_j_symbol = sp.Symbol(comp_j)
                
                # 检查comp_i的方程中是否有来自comp_j的流入项
                for term in terms:
                    # 跳过新感染项
                    if self._contains_susceptible_interaction(term):
                        continue
                    
                    if comp_j_symbol in term.free_symbols and comp_i_symbol not in term.free_symbols:
                        # 这是从comp_j到comp_i的转移项
                        linearized_term = self._linearize_term_at_dfe(term)
                        coeff = linearized_term.coeff(comp_j_symbol, 1)
                        if coeff is not None and coeff != 0:
                            V_matrix[i, j] = -coeff  # 转移项，取负号
                            print(f"  转移项 {comp_j}->{comp_i}: {term} -> V[{i},{j}] = {-coeff}")
    
    #这个函数是构造F矩阵
    def _construct_F_matrix(self, F_matrix):
        """
        构造F矩阵：新感染项
        """
        print("\n构造F矩阵...")
        
        # 寻找新感染项 - 通常在第一个感染仓室的方程中
        for i, comp_i in enumerate(self.infected_compartments):
            equation = self.model_equations[comp_i]
            
            print(f"\n分析仓室 {comp_i} 的新感染项: d{comp_i}/dt = {equation}")
            
            terms = sp.Add.make_args(equation)
            
            for term in terms:
                # 检查是否为新感染项（包含易感仓室）
                if self._contains_susceptible_interaction(term):
                    print(f"  发现新感染项: {term}")
                    
                    # 线性化
                    linearized_term = self._linearize_term_at_dfe(term)
                    print(f"    线性化后: {linearized_term}")
                    
                    # 对于新感染项，我们需要提取传播矩阵的系数
                    # 新感染项通常形如 β*S*I/N，线性化后成为 β*I（当S=N时）
                    # 我们需要将其放入正确的F矩阵位置
                    
                    # 检查这个新感染项中涉及的感染仓室
                    infected_symbols_in_term = [sp.Symbol(comp) for comp in self.infected_compartments 
                                               if sp.Symbol(comp) in linearized_term.free_symbols]
                    
                    if len(infected_symbols_in_term) == 1:
                        # 标准情况：新感染项只涉及一个感染仓室
                        source_symbol = infected_symbols_in_term[0]
                        source_index = self.infected_compartments.index(str(source_symbol))
                        
                        # 提取传播率（去掉感染仓室符号）
                        transmission_rate = linearized_term.coeff(source_symbol, 1)
                        if transmission_rate is None:
                            transmission_rate = 0
                        
                        F_matrix[i, source_index] = transmission_rate
                        print(f"    -> F[{i},{source_index}] = {transmission_rate}")
                    
                    elif len(infected_symbols_in_term) == 0:
                        # 线性化后不包含感染仓室符号，这是常数传播率
                        # 通常放在F[i,i]位置
                        F_matrix[i, i] = linearized_term
                        print(f"    -> F[{i},{i}] = {linearized_term} (常数传播率)")                    
    #这个函数是线性化单个项
    #线性化的步骤是先将非感染仓室替换为其平衡点值
    #然后对感染仓室进行泰勒展开，保留到一次项   
    def _linearize_term_at_dfe(self, term: sp.Expr) -> sp.Expr:
        """
        将单个项在无病平衡点处线性化
        """
        linearized = term
        
        # 替换非感染仓室为其平衡点值
        for comp, equilibrium_val in self.disease_free_equilibrium.items():
            if comp not in self.infected_compartments:
                comp_symbol = sp.Symbol(comp)
                linearized = linearized.subs(comp_symbol, equilibrium_val)
        
        # 对感染仓室进行泰勒展开
        infected_symbols = [sp.Symbol(comp) for comp in self.infected_compartments]
        for sym in infected_symbols:
            linearized = sp.series(linearized, sym, 0, n=2).removeO()
        
        return linearized
    


    #这个函数是检查项是否包含易感仓室的相互作用
    def _contains_susceptible_interaction(self, term: sp.Expr) -> bool:
        """
        判断项是否包含与易感仓室的相互作用
        """
        term_symbols = term.free_symbols
        
        # 检查是否同时包含感染仓室和易感仓室
        has_infected = any(sp.Symbol(comp) in term_symbols for comp in self.infected_compartments)
        has_susceptible = any(sp.Symbol(comp) in term_symbols 
                            for comp in self.all_compartments 
                            if comp not in self.infected_compartments)
        
        return has_infected and has_susceptible
    
    
    def compute_R0(self):
        """
        计算基本再生数 R₀ = ρ(FV⁻¹)
        """
        if self.F_matrix is None or self.V_matrix is None:
            self.extract_F_V_matrices()
        
        print("\n" + "=" * 60)
        print("正在计算基本再生数...")
        
        # 计算V矩阵的逆
        print("计算V⁻¹...")
        try:
            V_inv = self.V_matrix.inv()
            print("V⁻¹ =")
            pprint(V_inv)
        except Exception as e:
            print(f"错误：V矩阵不可逆！{e}")
            return None
        
        # 计算下一代矩阵
        print("\n计算下一代矩阵 NGM = F × V⁻¹...")
        self.NGM = simplify(self.F_matrix * V_inv)
        print("NGM =")
        pprint(self.NGM)
        
        # 计算特征值
        print("\n计算特征值...")
        if self.NGM.shape == (1, 1):
            # 1x1矩阵的情况
            self.R0 = self.NGM[0, 0]
        else:
            # 多维矩阵的情况
            eigenvals = self.NGM.eigenvals()
            print("特征值:")
            for i, (eigval, mult) in enumerate(eigenvals.items()):
                print(f"λ_{i+1} = {eigval} (重数: {mult})")
            
            # R₀是最大特征值（谱半径）
            # 排除零特征值，选择最大的非零特征值
            non_zero_eigenvals = [eigval for eigval in eigenvals.keys() if eigval != 0]
            
            if non_zero_eigenvals:
                if len(non_zero_eigenvals) == 1:
                    self.R0 = non_zero_eigenvals[0]
                else:
                    # 对于多个非零特征值，选择正的那个（谱半径）
                    try:
                        # 基本再生数应该是正值，选择正的特征值
                        positive_eigenvals = [eigval for eigval in non_zero_eigenvals 
                                            if complex(eigval.evalf()).real > 0]
                        if positive_eigenvals:
                            # 选择最大的正特征值
                            max_eigval = max(positive_eigenvals, key=lambda x: complex(x.evalf()).real)
                            self.R0 = max_eigval
                        else:
                            # 如果没有正特征值，取绝对值最大的
                            max_eigval = max(non_zero_eigenvals, key=lambda x: abs(complex(x.evalf())))
                            self.R0 = sp.Abs(max_eigval)
                    except:
                        # 如果数值比较失败，取第一个非零特征值的绝对值
                        self.R0 = sp.Abs(non_zero_eigenvals[0])
            else:
                # 如果所有特征值都是零，R₀ = 0
                self.R0 = 0
        
        # 简化R₀表达式
        self.R0 = simplify(self.R0)
        
        print(f"\n基本再生数 R₀:")
        pprint(self.R0)
        
        return self.R0
    
    def analyze_results(self, latex_output: bool = True):
        """
        分析并输出结果
        """
        print("\n" + "=" * 60)
        print("完整分析结果")
        print("=" * 60)
        
        print("\n1. 模型方程组:")
        for comp, eq in self.model_equations.items():
            print(f"   d{comp}/dt = {eq}")
        
        print(f"\n2. 感染仓室: {self.infected_compartments}")
        
        print(f"\n3. F矩阵 (新感染项):")
        pprint(self.F_matrix)
        
        print(f"\n4. V矩阵 (转移项):")
        pprint(self.V_matrix)
        
        print(f"\n5. 下一代矩阵 NGM = FV⁻¹:")
        pprint(self.NGM)
        
        print(f"\n6. 基本再生数 R₀:")
        pprint(self.R0)
        
        if latex_output:
            print(f"\n7. LaTeX格式:")
            print(f"   R₀ = {latex(self.R0)}")
        return self

#这个函数是运行示例的辅助函数 
#参数分别是模型名称（name自己取名即可） 微分方程组（equations） 感染相关仓室（infected_compartments） 无病平衡点（equilibrium）
def run_example(name, equations, infected_compartments, equilibrium=None):
    """
    运行示例的辅助函数，简化重复代码
    参数:
    name: 模型名称
    equations: 微分方程组
    infected_compartments: 感染相关仓室 列表
    equilibrium: 无病平衡点 字典（可选，如果不提供则自动设置）
    该函数会创建计算器实例，定义模型，设置平衡点，计算R₀，并分析结果
    只需要传入模型相关参数即可  
    适用于所有示例模型
    例如：
    - 手动设置平衡点：run_example("示例1：SIR模型", equations, ['I'], {'S': N, 'I': 0, 'R': 0})
    - 自动设置平衡点：run_example("示例1：SIR模型", equations, ['I'])
    """
    print(f"\n\n{name}")
    print("=" * 60)
    
    # 创建计算器并分析
    calc = SmartR0Calculator()
    calc.infectious_model(equations, infected_compartments=infected_compartments)
    calc.set_disease_free_equilibrium(equilibrium)
    calc.compute_R0()
    calc.analyze_results()
