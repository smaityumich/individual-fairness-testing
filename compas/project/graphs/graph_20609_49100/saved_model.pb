
Ķ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.0-dev202008102v1.12.1-38915-gfe968502a98¶é
x
layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namelayer-1/kernel
q
"layer-1/kernel/Read/ReadVariableOpReadVariableOplayer-1/kernel*
_output_shapes

:2*
dtype0
p
layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namelayer-1/bias
i
 layer-1/bias/Read/ReadVariableOpReadVariableOplayer-1/bias*
_output_shapes
:2*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:2*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:*
dtype0

NoOpNoOp
ą
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
i

Layers
		model

regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
3
#
0
1
2
3
4
­

layers
regularization_losses
non_trainable_variables
layer_metrics
metrics
layer_regularization_losses
trainable_variables
	variables
 

0
1
2
Ē
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
3
#
0
1
2
3
4
­

layers

regularization_losses
 non_trainable_variables
!layer_metrics
"metrics
#layer_regularization_losses
trainable_variables
	variables
TR
VARIABLE_VALUElayer-1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer-1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEoutput/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEoutput/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEVariable&variables/4/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
 
 
 
Y
w
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

kernel
bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
 

0
1
2
3
#
0
1
2
3
4
­

0layers
regularization_losses
1non_trainable_variables
2layer_metrics
3metrics
4layer_regularization_losses
trainable_variables
	variables

0
1
2
	3

0
 
 
 
 
 

0
­

5layers
$regularization_losses
6non_trainable_variables
7layer_metrics
8metrics
9layer_regularization_losses
%trainable_variables
&	variables
 

0
1

0
1
­

:layers
(regularization_losses
;non_trainable_variables
<layer_metrics
=metrics
>layer_regularization_losses
)trainable_variables
*	variables
 

0
1

0
1
­

?layers
,regularization_losses
@non_trainable_variables
Alayer_metrics
Bmetrics
Clayer_regularization_losses
-trainable_variables
.	variables

0
1
2

0
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_5Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3167684
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
É
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer-1/kernel/Read/ReadVariableOp layer-1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3168172
ą
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer-1/kernellayer-1/biasoutput/kerneloutput/biasVariable*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3168197Łæ
¦	

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167431
x
sequential_4_3167419
sequential_4_3167421
sequential_4_3167423
sequential_4_3167425
sequential_4_3167427
identity¢$sequential_4/StatefulPartitionedCallī
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallxsequential_4_3167419sequential_4_3167421sequential_4_3167423sequential_4_3167425sequential_4_3167427*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_31673242&
$sequential_4/StatefulPartitionedCallØ
IdentityIdentity-sequential_4/StatefulPartitionedCall:output:0%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ä
±
4__inference_classifier_graph_4_layer_call_fn_3167915
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ź
±
.__inference_functional_9_layer_call_fn_3167622
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_9_layer_call_and_return_conditional_losses_31675942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
Ä
±
4__inference_classifier_graph_4_layer_call_fn_3167930
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex


µ
I__inference_functional_9_layer_call_and_return_conditional_losses_3167594

inputs
classifier_graph_4_3167582
classifier_graph_4_3167584
classifier_graph_4_3167586
classifier_graph_4_3167588
classifier_graph_4_3167590
identity¢*classifier_graph_4/StatefulPartitionedCall£
*classifier_graph_4/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_4_3167582classifier_graph_4_3167584classifier_graph_4_3167586classifier_graph_4_3167588classifier_graph_4_3167590*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31675032,
*classifier_graph_4/StatefulPartitionedCall“
IdentityIdentity3classifier_graph_4/StatefulPartitionedCall:output:0+^classifier_graph_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2X
*classifier_graph_4/StatefulPartitionedCall*classifier_graph_4/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ę 

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167900
xC
?sequential_4_project_4_matrix_transpose_readvariableop_resource7
3sequential_4_layer_1_matmul_readvariableop_resource8
4sequential_4_layer_1_biasadd_readvariableop_resource6
2sequential_4_output_matmul_readvariableop_resource7
3sequential_4_output_biasadd_readvariableop_resource
identityš
6sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_4/project_4/matrix_transpose/ReadVariableOpĮ
6sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_4/project_4/matrix_transpose/transpose/perm
1sequential_4/project_4/matrix_transpose/transpose	Transpose>sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0?sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_4/project_4/matrix_transpose/transpose“
sequential_4/project_4/matmulMatMulx5sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/matmulą
.sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_4/project_4/matmul_1/ReadVariableOpß
sequential_4/project_4/matmul_1MatMul'sequential_4/project_4/matmul:product:06sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_4/project_4/matmul_1
sequential_4/project_4/subSubx)sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/subĢ
*sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_4/layer-1/MatMul/ReadVariableOpŹ
sequential_4/layer-1/MatMulMatMulsequential_4/project_4/sub:z:02sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/MatMulĖ
+sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
sequential_4/layer-1/BiasAddBiasAdd%sequential_4/layer-1/MatMul:product:03sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/BiasAdd
sequential_4/layer-1/ReluRelu%sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/ReluÉ
)sequential_4/output/MatMul/ReadVariableOpReadVariableOp2sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_4/output/MatMul/ReadVariableOpŠ
sequential_4/output/MatMulMatMul'sequential_4/layer-1/Relu:activations:01sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/MatMulČ
*sequential_4/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_4/output/BiasAdd/ReadVariableOpŃ
sequential_4/output/BiasAddBiasAdd$sequential_4/output/MatMul:product:02sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/BiasAdd
sequential_4/output/SoftmaxSoftmax$sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/Softmaxy
IdentityIdentity%sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ś
}
(__inference_output_layer_call_fn_3168134

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31671972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ž 

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167818
input_1C
?sequential_4_project_4_matrix_transpose_readvariableop_resource7
3sequential_4_layer_1_matmul_readvariableop_resource8
4sequential_4_layer_1_biasadd_readvariableop_resource6
2sequential_4_output_matmul_readvariableop_resource7
3sequential_4_output_biasadd_readvariableop_resource
identityš
6sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_4/project_4/matrix_transpose/ReadVariableOpĮ
6sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_4/project_4/matrix_transpose/transpose/perm
1sequential_4/project_4/matrix_transpose/transpose	Transpose>sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0?sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_4/project_4/matrix_transpose/transposeŗ
sequential_4/project_4/matmulMatMulinput_15sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/matmulą
.sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_4/project_4/matmul_1/ReadVariableOpß
sequential_4/project_4/matmul_1MatMul'sequential_4/project_4/matmul:product:06sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_4/project_4/matmul_1„
sequential_4/project_4/subSubinput_1)sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/subĢ
*sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_4/layer-1/MatMul/ReadVariableOpŹ
sequential_4/layer-1/MatMulMatMulsequential_4/project_4/sub:z:02sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/MatMulĖ
+sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
sequential_4/layer-1/BiasAddBiasAdd%sequential_4/layer-1/MatMul:product:03sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/BiasAdd
sequential_4/layer-1/ReluRelu%sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/ReluÉ
)sequential_4/output/MatMul/ReadVariableOpReadVariableOp2sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_4/output/MatMul/ReadVariableOpŠ
sequential_4/output/MatMulMatMul'sequential_4/layer-1/Relu:activations:01sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/MatMulČ
*sequential_4/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_4/output/BiasAdd/ReadVariableOpŃ
sequential_4/output/BiasAddBiasAdd$sequential_4/output/MatMul:product:02sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/BiasAdd
sequential_4/output/SoftmaxSoftmax$sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/Softmaxy
IdentityIdentity%sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
©
¬
D__inference_layer-1_layer_call_and_return_conditional_losses_3167150

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
*
š
I__inference_functional_9_layer_call_and_return_conditional_losses_3167736

inputsV
Rclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_4_sequential_4_output_matmul_readvariableop_resourceJ
Fclassifier_graph_4_sequential_4_output_biasadd_readvariableop_resource
identity©
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02K
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOpē
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/permé
Dclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose	TransposeQclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2F
Dclassifier_graph_4/sequential_4/project_4/matrix_transpose/transposeņ
0classifier_graph_4/sequential_4/project_4/matmulMatMulinputsHclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’22
0classifier_graph_4/sequential_4/project_4/matmul
Aclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02C
Aclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOp«
2classifier_graph_4/sequential_4/project_4/matmul_1MatMul:classifier_graph_4/sequential_4/project_4/matmul:product:0Iclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’24
2classifier_graph_4/sequential_4/project_4/matmul_1Ż
-classifier_graph_4/sequential_4/project_4/subSubinputs<classifier_graph_4/sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-classifier_graph_4/sequential_4/project_4/sub
=classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02?
=classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOp
.classifier_graph_4/sequential_4/layer-1/MatMulMatMul1classifier_graph_4/sequential_4/project_4/sub:z:0Eclassifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’220
.classifier_graph_4/sequential_4/layer-1/MatMul
>classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOp”
/classifier_graph_4/sequential_4/layer-1/BiasAddBiasAdd8classifier_graph_4/sequential_4/layer-1/MatMul:product:0Fclassifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’221
/classifier_graph_4/sequential_4/layer-1/BiasAddŠ
,classifier_graph_4/sequential_4/layer-1/ReluRelu8classifier_graph_4/sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22.
,classifier_graph_4/sequential_4/layer-1/Relu
<classifier_graph_4/sequential_4/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_4_sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_4/sequential_4/output/MatMul/ReadVariableOp
-classifier_graph_4/sequential_4/output/MatMulMatMul:classifier_graph_4/sequential_4/layer-1/Relu:activations:0Dclassifier_graph_4/sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-classifier_graph_4/sequential_4/output/MatMul
=classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_4_sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOp
.classifier_graph_4/sequential_4/output/BiasAddBiasAdd7classifier_graph_4/sequential_4/output/MatMul:product:0Eclassifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’20
.classifier_graph_4/sequential_4/output/BiasAddÖ
.classifier_graph_4/sequential_4/output/SoftmaxSoftmax7classifier_graph_4/sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’20
.classifier_graph_4/sequential_4/output/Softmax
IdentityIdentity8classifier_graph_4/sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü
~
)__inference_layer-1_layer_call_fn_3168114

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_31671502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž 

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167792
input_1C
?sequential_4_project_4_matrix_transpose_readvariableop_resource7
3sequential_4_layer_1_matmul_readvariableop_resource8
4sequential_4_layer_1_biasadd_readvariableop_resource6
2sequential_4_output_matmul_readvariableop_resource7
3sequential_4_output_biasadd_readvariableop_resource
identityš
6sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_4/project_4/matrix_transpose/ReadVariableOpĮ
6sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_4/project_4/matrix_transpose/transpose/perm
1sequential_4/project_4/matrix_transpose/transpose	Transpose>sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0?sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_4/project_4/matrix_transpose/transposeŗ
sequential_4/project_4/matmulMatMulinput_15sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/matmulą
.sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_4/project_4/matmul_1/ReadVariableOpß
sequential_4/project_4/matmul_1MatMul'sequential_4/project_4/matmul:product:06sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_4/project_4/matmul_1„
sequential_4/project_4/subSubinput_1)sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/subĢ
*sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_4/layer-1/MatMul/ReadVariableOpŹ
sequential_4/layer-1/MatMulMatMulsequential_4/project_4/sub:z:02sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/MatMulĖ
+sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
sequential_4/layer-1/BiasAddBiasAdd%sequential_4/layer-1/MatMul:product:03sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/BiasAdd
sequential_4/layer-1/ReluRelu%sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/ReluÉ
)sequential_4/output/MatMul/ReadVariableOpReadVariableOp2sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_4/output/MatMul/ReadVariableOpŠ
sequential_4/output/MatMulMatMul'sequential_4/layer-1/Relu:activations:01sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/MatMulČ
*sequential_4/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_4/output/BiasAdd/ReadVariableOpŃ
sequential_4/output/BiasAddBiasAdd$sequential_4/output/MatMul:product:02sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/BiasAdd
sequential_4/output/SoftmaxSoftmax$sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/Softmaxy
IdentityIdentity%sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
š
Ł
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168038
project_4_input6
2project_4_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityÉ
)project_4/matrix_transpose/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_4/matrix_transpose/ReadVariableOp§
)project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_4/matrix_transpose/transpose/permé
$project_4/matrix_transpose/transpose	Transpose1project_4/matrix_transpose/ReadVariableOp:value:02project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_4/matrix_transpose/transpose
project_4/matmulMatMulproject_4_input(project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul¹
!project_4/matmul_1/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02#
!project_4/matmul_1/ReadVariableOp«
project_4/matmul_1MatMulproject_4/matmul:product:0)project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul_1
project_4/subSubproject_4_inputproject_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/sub„
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp
layer-1/MatMulMatMulproject_4/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/MatMul¤
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp”
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/Relu¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/MatMul”
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameproject_4_input

ø
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167324

inputs
project_4_3167310
layer_1_3167313
layer_1_3167315
output_3167318
output_3167320
identity¢layer-1/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!project_4/StatefulPartitionedCall
!project_4/StatefulPartitionedCallStatefulPartitionedCallinputsproject_4_3167310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_project_4_layer_call_and_return_conditional_losses_31671062#
!project_4/StatefulPartitionedCall¶
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_4/StatefulPartitionedCall:output:0layer_1_3167313layer_1_3167315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_31671502!
layer-1/StatefulPartitionedCallÆ
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_3167318output_3167320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31671972 
output/StatefulPartitionedCallā
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_4/StatefulPartitionedCall!project_4/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
0

"__inference__wrapped_model_3167081
input_5c
_functional_9_classifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resourceW
Sfunctional_9_classifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resourceX
Tfunctional_9_classifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resourceV
Rfunctional_9_classifier_graph_4_sequential_4_output_matmul_readvariableop_resourceW
Sfunctional_9_classifier_graph_4_sequential_4_output_biasadd_readvariableop_resource
identityŠ
Vfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp_functional_9_classifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02X
Vfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOp
Vfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2X
Vfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/perm
Qfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose	Transpose^functional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0_functional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2S
Qfunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose
=functional_9/classifier_graph_4/sequential_4/project_4/matmulMatMulinput_5Ufunctional_9/classifier_graph_4/sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2?
=functional_9/classifier_graph_4/sequential_4/project_4/matmulĄ
Nfunctional_9/classifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp_functional_9_classifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02P
Nfunctional_9/classifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOpß
?functional_9/classifier_graph_4/sequential_4/project_4/matmul_1MatMulGfunctional_9/classifier_graph_4/sequential_4/project_4/matmul:product:0Vfunctional_9/classifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2A
?functional_9/classifier_graph_4/sequential_4/project_4/matmul_1
:functional_9/classifier_graph_4/sequential_4/project_4/subSubinput_5Ifunctional_9/classifier_graph_4/sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2<
:functional_9/classifier_graph_4/sequential_4/project_4/sub¬
Jfunctional_9/classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOpSfunctional_9_classifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02L
Jfunctional_9/classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOpŹ
;functional_9/classifier_graph_4/sequential_4/layer-1/MatMulMatMul>functional_9/classifier_graph_4/sequential_4/project_4/sub:z:0Rfunctional_9/classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22=
;functional_9/classifier_graph_4/sequential_4/layer-1/MatMul«
Kfunctional_9/classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOpTfunctional_9_classifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02M
Kfunctional_9/classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
<functional_9/classifier_graph_4/sequential_4/layer-1/BiasAddBiasAddEfunctional_9/classifier_graph_4/sequential_4/layer-1/MatMul:product:0Sfunctional_9/classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22>
<functional_9/classifier_graph_4/sequential_4/layer-1/BiasAdd÷
9functional_9/classifier_graph_4/sequential_4/layer-1/ReluReluEfunctional_9/classifier_graph_4/sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22;
9functional_9/classifier_graph_4/sequential_4/layer-1/Relu©
Ifunctional_9/classifier_graph_4/sequential_4/output/MatMul/ReadVariableOpReadVariableOpRfunctional_9_classifier_graph_4_sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02K
Ifunctional_9/classifier_graph_4/sequential_4/output/MatMul/ReadVariableOpŠ
:functional_9/classifier_graph_4/sequential_4/output/MatMulMatMulGfunctional_9/classifier_graph_4/sequential_4/layer-1/Relu:activations:0Qfunctional_9/classifier_graph_4/sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2<
:functional_9/classifier_graph_4/sequential_4/output/MatMulØ
Jfunctional_9/classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOpReadVariableOpSfunctional_9_classifier_graph_4_sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jfunctional_9/classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOpŃ
;functional_9/classifier_graph_4/sequential_4/output/BiasAddBiasAddDfunctional_9/classifier_graph_4/sequential_4/output/MatMul:product:0Rfunctional_9/classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2=
;functional_9/classifier_graph_4/sequential_4/output/BiasAddż
;functional_9/classifier_graph_4/sequential_4/output/SoftmaxSoftmaxDfunctional_9/classifier_graph_4/sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2=
;functional_9/classifier_graph_4/sequential_4/output/Softmax
IdentityIdentityEfunctional_9/classifier_graph_4/sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
Ē
°
.__inference_sequential_4_layer_call_fn_3168012

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_31673242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ź
±
.__inference_functional_9_layer_call_fn_3167652
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_9_layer_call_and_return_conditional_losses_31676392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
Ē
°
.__inference_functional_9_layer_call_fn_3167766

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_9_layer_call_and_return_conditional_losses_31676392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž


F__inference_project_4_layer_call_and_return_conditional_losses_3167106
x,
(matrix_transpose_readvariableop_resource
identity«
matrix_transpose/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02!
matrix_transpose/ReadVariableOp
matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2!
matrix_transpose/transpose/permĮ
matrix_transpose/transpose	Transpose'matrix_transpose/ReadVariableOp:value:0(matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2
matrix_transpose/transposeo
matmulMatMulxmatrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
matmul
matmul_1/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02
matmul_1/ReadVariableOp
matmul_1MatMulmatmul:product:0matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2

matmul_1Z
subSubxmatmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’::J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
©
¬
D__inference_layer-1_layer_call_and_return_conditional_losses_3168105

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
·
4__inference_classifier_graph_4_layer_call_fn_3167833
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ė
Š
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167982

inputs6
2project_4_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityÉ
)project_4/matrix_transpose/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_4/matrix_transpose/ReadVariableOp§
)project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_4/matrix_transpose/transpose/permé
$project_4/matrix_transpose/transpose	Transpose1project_4/matrix_transpose/ReadVariableOp:value:02project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_4/matrix_transpose/transpose
project_4/matmulMatMulinputs(project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul¹
!project_4/matmul_1/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02#
!project_4/matmul_1/ReadVariableOp«
project_4/matmul_1MatMulproject_4/matmul:product:0)project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul_1}
project_4/subSubinputsproject_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/sub„
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp
layer-1/MatMulMatMulproject_4/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/MatMul¤
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp”
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/Relu¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/MatMul”
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ģ
 __inference__traced_save_3168172
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop'
#savev2_variable_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_84eb4f313ffe4405bd220e103008f590/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.: :2:2:2::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
Ē
°
.__inference_functional_9_layer_call_fn_3167751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_9_layer_call_and_return_conditional_losses_31675942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


¶
I__inference_functional_9_layer_call_and_return_conditional_losses_3167576
input_5
classifier_graph_4_3167564
classifier_graph_4_3167566
classifier_graph_4_3167568
classifier_graph_4_3167570
classifier_graph_4_3167572
identity¢*classifier_graph_4/StatefulPartitionedCall¤
*classifier_graph_4/StatefulPartitionedCallStatefulPartitionedCallinput_5classifier_graph_4_3167564classifier_graph_4_3167566classifier_graph_4_3167568classifier_graph_4_3167570classifier_graph_4_3167572*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312,
*classifier_graph_4/StatefulPartitionedCall“
IdentityIdentity3classifier_graph_4/StatefulPartitionedCall:output:0+^classifier_graph_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2X
*classifier_graph_4/StatefulPartitionedCall*classifier_graph_4/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
Ē
°
.__inference_sequential_4_layer_call_fn_3167997

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_31672772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ø
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167277

inputs
project_4_3167263
layer_1_3167266
layer_1_3167268
output_3167271
output_3167273
identity¢layer-1/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!project_4/StatefulPartitionedCall
!project_4/StatefulPartitionedCallStatefulPartitionedCallinputsproject_4_3167263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_project_4_layer_call_and_return_conditional_losses_31671062#
!project_4/StatefulPartitionedCall¶
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_4/StatefulPartitionedCall:output:0layer_1_3167266layer_1_3167268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_31671502!
layer-1/StatefulPartitionedCallÆ
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_3167271output_3167273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31671972 
output/StatefulPartitionedCallā
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_4/StatefulPartitionedCall!project_4/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


¶
I__inference_functional_9_layer_call_and_return_conditional_losses_3167561
input_5
classifier_graph_4_3167549
classifier_graph_4_3167551
classifier_graph_4_3167553
classifier_graph_4_3167555
classifier_graph_4_3167557
identity¢*classifier_graph_4/StatefulPartitionedCall¤
*classifier_graph_4/StatefulPartitionedCallStatefulPartitionedCallinput_5classifier_graph_4_3167549classifier_graph_4_3167551classifier_graph_4_3167553classifier_graph_4_3167555classifier_graph_4_3167557*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31675032,
*classifier_graph_4/StatefulPartitionedCall“
IdentityIdentity3classifier_graph_4/StatefulPartitionedCall:output:0+^classifier_graph_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2X
*classifier_graph_4/StatefulPartitionedCall*classifier_graph_4/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
Ö
·
4__inference_classifier_graph_4_layer_call_fn_3167848
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ę 

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167874
xC
?sequential_4_project_4_matrix_transpose_readvariableop_resource7
3sequential_4_layer_1_matmul_readvariableop_resource8
4sequential_4_layer_1_biasadd_readvariableop_resource6
2sequential_4_output_matmul_readvariableop_resource7
3sequential_4_output_biasadd_readvariableop_resource
identityš
6sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_4/project_4/matrix_transpose/ReadVariableOpĮ
6sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_4/project_4/matrix_transpose/transpose/perm
1sequential_4/project_4/matrix_transpose/transpose	Transpose>sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0?sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_4/project_4/matrix_transpose/transpose“
sequential_4/project_4/matmulMatMulx5sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/matmulą
.sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_4/project_4/matmul_1/ReadVariableOpß
sequential_4/project_4/matmul_1MatMul'sequential_4/project_4/matmul:product:06sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_4/project_4/matmul_1
sequential_4/project_4/subSubx)sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/subĢ
*sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_4/layer-1/MatMul/ReadVariableOpŹ
sequential_4/layer-1/MatMulMatMulsequential_4/project_4/sub:z:02sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/MatMulĖ
+sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
sequential_4/layer-1/BiasAddBiasAdd%sequential_4/layer-1/MatMul:product:03sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/BiasAdd
sequential_4/layer-1/ReluRelu%sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/ReluÉ
)sequential_4/output/MatMul/ReadVariableOpReadVariableOp2sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_4/output/MatMul/ReadVariableOpŠ
sequential_4/output/MatMulMatMul'sequential_4/layer-1/Relu:activations:01sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/MatMulČ
*sequential_4/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_4/output/BiasAdd/ReadVariableOpŃ
sequential_4/output/BiasAddBiasAdd$sequential_4/output/MatMul:product:02sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/BiasAdd
sequential_4/output/SoftmaxSoftmax$sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/Softmaxy
IdentityIdentity%sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
š
Ł
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168064
project_4_input6
2project_4_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityÉ
)project_4/matrix_transpose/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_4/matrix_transpose/ReadVariableOp§
)project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_4/matrix_transpose/transpose/permé
$project_4/matrix_transpose/transpose	Transpose1project_4/matrix_transpose/ReadVariableOp:value:02project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_4/matrix_transpose/transpose
project_4/matmulMatMulproject_4_input(project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul¹
!project_4/matmul_1/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02#
!project_4/matmul_1/ReadVariableOp«
project_4/matmul_1MatMulproject_4/matmul:product:0)project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul_1
project_4/subSubproject_4_inputproject_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/sub„
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp
layer-1/MatMulMatMulproject_4/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/MatMul¤
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp”
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/Relu¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/MatMul”
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameproject_4_input
ó
å
#__inference__traced_restore_3168197
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias
assignvariableop_4_variable

identity_6¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slicesÉ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2„
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4 
AssignVariableOp_4AssignVariableOpassignvariableop_4_variableIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpĻ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5Į

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


µ
I__inference_functional_9_layer_call_and_return_conditional_losses_3167639

inputs
classifier_graph_4_3167627
classifier_graph_4_3167629
classifier_graph_4_3167631
classifier_graph_4_3167633
classifier_graph_4_3167635
identity¢*classifier_graph_4/StatefulPartitionedCall£
*classifier_graph_4/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_4_3167627classifier_graph_4_3167629classifier_graph_4_3167631classifier_graph_4_3167633classifier_graph_4_3167635*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_31674312,
*classifier_graph_4/StatefulPartitionedCall“
IdentityIdentity3classifier_graph_4/StatefulPartitionedCall:output:0+^classifier_graph_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::2X
*classifier_graph_4/StatefulPartitionedCall*classifier_graph_4/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ā
¹
.__inference_sequential_4_layer_call_fn_3168079
project_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallproject_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_31672772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameproject_4_input
°
«
C__inference_output_layer_call_and_return_conditional_losses_3168125

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’2:::O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
±
l
+__inference_project_4_layer_call_fn_3167121
x
unknown
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_project_4_layer_call_and_return_conditional_losses_31671062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
°
«
C__inference_output_layer_call_and_return_conditional_losses_3167197

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’2:::O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ā
¹
.__inference_sequential_4_layer_call_fn_3168094
project_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallproject_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_31673242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameproject_4_input

Ø
%__inference_signature_wrapper_3167684
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_31670812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_5
ę 

O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167503
xC
?sequential_4_project_4_matrix_transpose_readvariableop_resource7
3sequential_4_layer_1_matmul_readvariableop_resource8
4sequential_4_layer_1_biasadd_readvariableop_resource6
2sequential_4_output_matmul_readvariableop_resource7
3sequential_4_output_biasadd_readvariableop_resource
identityš
6sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_4/project_4/matrix_transpose/ReadVariableOpĮ
6sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_4/project_4/matrix_transpose/transpose/perm
1sequential_4/project_4/matrix_transpose/transpose	Transpose>sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0?sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_4/project_4/matrix_transpose/transpose“
sequential_4/project_4/matmulMatMulx5sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/matmulą
.sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOp?sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_4/project_4/matmul_1/ReadVariableOpß
sequential_4/project_4/matmul_1MatMul'sequential_4/project_4/matmul:product:06sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_4/project_4/matmul_1
sequential_4/project_4/subSubx)sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/project_4/subĢ
*sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_4/layer-1/MatMul/ReadVariableOpŹ
sequential_4/layer-1/MatMulMatMulsequential_4/project_4/sub:z:02sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/MatMulĖ
+sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_4/layer-1/BiasAdd/ReadVariableOpÕ
sequential_4/layer-1/BiasAddBiasAdd%sequential_4/layer-1/MatMul:product:03sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/BiasAdd
sequential_4/layer-1/ReluRelu%sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
sequential_4/layer-1/ReluÉ
)sequential_4/output/MatMul/ReadVariableOpReadVariableOp2sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_4/output/MatMul/ReadVariableOpŠ
sequential_4/output/MatMulMatMul'sequential_4/layer-1/Relu:activations:01sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/MatMulČ
*sequential_4/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_4/output/BiasAdd/ReadVariableOpŃ
sequential_4/output/BiasAddBiasAdd$sequential_4/output/MatMul:product:02sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/BiasAdd
sequential_4/output/SoftmaxSoftmax$sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_4/output/Softmaxy
IdentityIdentity%sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::J F
'
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ė
Š
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167956

inputs6
2project_4_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityÉ
)project_4/matrix_transpose/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_4/matrix_transpose/ReadVariableOp§
)project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_4/matrix_transpose/transpose/permé
$project_4/matrix_transpose/transpose	Transpose1project_4/matrix_transpose/ReadVariableOp:value:02project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_4/matrix_transpose/transpose
project_4/matmulMatMulinputs(project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul¹
!project_4/matmul_1/ReadVariableOpReadVariableOp2project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02#
!project_4/matmul_1/ReadVariableOp«
project_4/matmul_1MatMulproject_4/matmul:product:0)project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/matmul_1}
project_4/subSubinputsproject_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
project_4/sub„
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp
layer-1/MatMulMatMulproject_4/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/MatMul¤
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp”
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
layer-1/Relu¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/MatMul”
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
*
š
I__inference_functional_9_layer_call_and_return_conditional_losses_3167710

inputsV
Rclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_4_sequential_4_output_matmul_readvariableop_resourceJ
Fclassifier_graph_4_sequential_4_output_biasadd_readvariableop_resource
identity©
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02K
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOpē
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/permé
Dclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose	TransposeQclassifier_graph_4/sequential_4/project_4/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2F
Dclassifier_graph_4/sequential_4/project_4/matrix_transpose/transposeņ
0classifier_graph_4/sequential_4/project_4/matmulMatMulinputsHclassifier_graph_4/sequential_4/project_4/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’22
0classifier_graph_4/sequential_4/project_4/matmul
Aclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_4_sequential_4_project_4_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02C
Aclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOp«
2classifier_graph_4/sequential_4/project_4/matmul_1MatMul:classifier_graph_4/sequential_4/project_4/matmul:product:0Iclassifier_graph_4/sequential_4/project_4/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’24
2classifier_graph_4/sequential_4/project_4/matmul_1Ż
-classifier_graph_4/sequential_4/project_4/subSubinputs<classifier_graph_4/sequential_4/project_4/matmul_1:product:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-classifier_graph_4/sequential_4/project_4/sub
=classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_4_sequential_4_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02?
=classifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOp
.classifier_graph_4/sequential_4/layer-1/MatMulMatMul1classifier_graph_4/sequential_4/project_4/sub:z:0Eclassifier_graph_4/sequential_4/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’220
.classifier_graph_4/sequential_4/layer-1/MatMul
>classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_4_sequential_4_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOp”
/classifier_graph_4/sequential_4/layer-1/BiasAddBiasAdd8classifier_graph_4/sequential_4/layer-1/MatMul:product:0Fclassifier_graph_4/sequential_4/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’221
/classifier_graph_4/sequential_4/layer-1/BiasAddŠ
,classifier_graph_4/sequential_4/layer-1/ReluRelu8classifier_graph_4/sequential_4/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’22.
,classifier_graph_4/sequential_4/layer-1/Relu
<classifier_graph_4/sequential_4/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_4_sequential_4_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_4/sequential_4/output/MatMul/ReadVariableOp
-classifier_graph_4/sequential_4/output/MatMulMatMul:classifier_graph_4/sequential_4/layer-1/Relu:activations:0Dclassifier_graph_4/sequential_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-classifier_graph_4/sequential_4/output/MatMul
=classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_4_sequential_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOp
.classifier_graph_4/sequential_4/output/BiasAddBiasAdd7classifier_graph_4/sequential_4/output/MatMul:product:0Eclassifier_graph_4/sequential_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’20
.classifier_graph_4/sequential_4/output/BiasAddÖ
.classifier_graph_4/sequential_4/output/SoftmaxSoftmax7classifier_graph_4/sequential_4/output/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’20
.classifier_graph_4/sequential_4/output/Softmax
IdentityIdentity8classifier_graph_4/sequential_4/output/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"čL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default”
;
input_50
serving_default_input_5:0’’’’’’’’’F
classifier_graph_40
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:é¢
Ė

layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
D_default_save_signature
E__call__
*F&call_and_return_all_conditional_losses"Ū
_tf_keras_networkæ{"class_name": "Functional", "name": "functional_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["classifier_graph_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 7]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
é"ę
_tf_keras_input_layerĘ{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
ø

Layers
		model

regularization_losses
trainable_variables
	variables
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_modelų{"class_name": "ClassifierGraph", "name": "classifier_graph_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
Ź

layers
regularization_losses
non_trainable_variables
layer_metrics
metrics
layer_regularization_losses
trainable_variables
	variables
E__call__
D_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
Æ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"«
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_4_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 7]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
­

layers

regularization_losses
 non_trainable_variables
!layer_metrics
"metrics
#layer_regularization_losses
trainable_variables
	variables
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 :22layer-1/kernel
:22layer-1/bias
:22output/kernel
:2output/bias
:2Variable
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ä
w
$regularization_losses
%trainable_variables
&	variables
'	keras_api
L__call__
*M&call_and_return_all_conditional_losses"Ī
_tf_keras_layer“{"class_name": "Project", "name": "project_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 7]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ž

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
N__call__
*O&call_and_return_all_conditional_losses"¹
_tf_keras_layer{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 7]}}
ļ

kernel
bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"Ź
_tf_keras_layer°{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 50]}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
­

0layers
regularization_losses
1non_trainable_variables
2layer_metrics
3metrics
4layer_regularization_losses
trainable_variables
	variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
	3"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
­

5layers
$regularization_losses
6non_trainable_variables
7layer_metrics
8metrics
9layer_regularization_losses
%trainable_variables
&	variables
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

:layers
(regularization_losses
;non_trainable_variables
<layer_metrics
=metrics
>layer_regularization_losses
)trainable_variables
*	variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

?layers
,regularization_losses
@non_trainable_variables
Alayer_metrics
Bmetrics
Clayer_regularization_losses
-trainable_variables
.	variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ą2Ż
"__inference__wrapped_model_3167081¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_5’’’’’’’’’
2
.__inference_functional_9_layer_call_fn_3167751
.__inference_functional_9_layer_call_fn_3167622
.__inference_functional_9_layer_call_fn_3167652
.__inference_functional_9_layer_call_fn_3167766Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ņ2ļ
I__inference_functional_9_layer_call_and_return_conditional_losses_3167710
I__inference_functional_9_layer_call_and_return_conditional_losses_3167561
I__inference_functional_9_layer_call_and_return_conditional_losses_3167576
I__inference_functional_9_layer_call_and_return_conditional_losses_3167736Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_classifier_graph_4_layer_call_fn_3167930
4__inference_classifier_graph_4_layer_call_fn_3167848
4__inference_classifier_graph_4_layer_call_fn_3167833
4__inference_classifier_graph_4_layer_call_fn_3167915½
“²°
FullArgSpec/
args'$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167792
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167874
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167818
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167900½
“²°
FullArgSpec/
args'$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
4B2
%__inference_signature_wrapper_3167684input_5
2
.__inference_sequential_4_layer_call_fn_3168079
.__inference_sequential_4_layer_call_fn_3167997
.__inference_sequential_4_layer_call_fn_3168012
.__inference_sequential_4_layer_call_fn_3168094Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ņ2ļ
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168038
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167956
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167982
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168064Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ė2č
+__inference_project_4_layer_call_fn_3167121ø
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *¢
’’’’’’’’’
2
F__inference_project_4_layer_call_and_return_conditional_losses_3167106ø
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *¢
’’’’’’’’’
Ó2Š
)__inference_layer-1_layer_call_fn_3168114¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_layer-1_layer_call_and_return_conditional_losses_3168105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_output_layer_call_fn_3168134¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_output_layer_call_and_return_conditional_losses_3168125¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ©
"__inference__wrapped_model_31670810¢-
&¢#
!
input_5’’’’’’’’’
Ŗ "GŖD
B
classifier_graph_4,)
classifier_graph_4’’’’’’’’’»
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167792h8¢5
.¢+
!
input_1’’’’’’’’’
p 
p
Ŗ "%¢"

0’’’’’’’’’
 »
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167818h8¢5
.¢+
!
input_1’’’’’’’’’
p 
p 
Ŗ "%¢"

0’’’’’’’’’
 µ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167874b2¢/
(¢%

x’’’’’’’’’
p 
p
Ŗ "%¢"

0’’’’’’’’’
 µ
O__inference_classifier_graph_4_layer_call_and_return_conditional_losses_3167900b2¢/
(¢%

x’’’’’’’’’
p 
p 
Ŗ "%¢"

0’’’’’’’’’
 
4__inference_classifier_graph_4_layer_call_fn_3167833[8¢5
.¢+
!
input_1’’’’’’’’’
p 
p
Ŗ "’’’’’’’’’
4__inference_classifier_graph_4_layer_call_fn_3167848[8¢5
.¢+
!
input_1’’’’’’’’’
p 
p 
Ŗ "’’’’’’’’’
4__inference_classifier_graph_4_layer_call_fn_3167915U2¢/
(¢%

x’’’’’’’’’
p 
p
Ŗ "’’’’’’’’’
4__inference_classifier_graph_4_layer_call_fn_3167930U2¢/
(¢%

x’’’’’’’’’
p 
p 
Ŗ "’’’’’’’’’µ
I__inference_functional_9_layer_call_and_return_conditional_losses_3167561h8¢5
.¢+
!
input_5’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 µ
I__inference_functional_9_layer_call_and_return_conditional_losses_3167576h8¢5
.¢+
!
input_5’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 “
I__inference_functional_9_layer_call_and_return_conditional_losses_3167710g7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 “
I__inference_functional_9_layer_call_and_return_conditional_losses_3167736g7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
.__inference_functional_9_layer_call_fn_3167622[8¢5
.¢+
!
input_5’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_functional_9_layer_call_fn_3167652[8¢5
.¢+
!
input_5’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_functional_9_layer_call_fn_3167751Z7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_functional_9_layer_call_fn_3167766Z7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’¤
D__inference_layer-1_layer_call_and_return_conditional_losses_3168105\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’2
 |
)__inference_layer-1_layer_call_fn_3168114O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’2£
C__inference_output_layer_call_and_return_conditional_losses_3168125\/¢,
%¢"
 
inputs’’’’’’’’’2
Ŗ "%¢"

0’’’’’’’’’
 {
(__inference_output_layer_call_fn_3168134O/¢,
%¢"
 
inputs’’’’’’’’’2
Ŗ "’’’’’’’’’ 
F__inference_project_4_layer_call_and_return_conditional_losses_3167106V*¢'
 ¢

x’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 x
+__inference_project_4_layer_call_fn_3167121I*¢'
 ¢

x’’’’’’’’’
Ŗ "’’’’’’’’’“
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167956g7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 “
I__inference_sequential_4_layer_call_and_return_conditional_losses_3167982g7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ½
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168038p@¢=
6¢3
)&
project_4_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ½
I__inference_sequential_4_layer_call_and_return_conditional_losses_3168064p@¢=
6¢3
)&
project_4_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
.__inference_sequential_4_layer_call_fn_3167997Z7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_3168012Z7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_3168079c@¢=
6¢3
)&
project_4_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_3168094c@¢=
6¢3
)&
project_4_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’·
%__inference_signature_wrapper_3167684;¢8
¢ 
1Ŗ.
,
input_5!
input_5’’’’’’’’’"GŖD
B
classifier_graph_4,)
classifier_graph_4’’’’’’’’’